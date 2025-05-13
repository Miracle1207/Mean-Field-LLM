import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import numpy as np
from torch.amp import autocast


# 设置随机种子和设备
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def generate_candidates_with_log_probs(input_ids, top_k, model, tokenizer, max_length=50, top_p=0.95, temperature=0.9):
    """
    根据输入 prompt 生成 top_k 个候选序列，并计算其 log 概率。
    使用 autocast 以减小显存占用。
    """
    with autocast("cuda"):
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=min(max_length, 50),
            num_return_sequences=min(top_k, 3),
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            output_scores=True,
            return_dict_in_generate=True
        )
    
    candidates = [
        tokenizer.decode(output[len(input_ids[0]):], skip_special_tokens=True)
        for output in outputs.sequences
    ]
    
    log_probs = []
    for sequence_idx, sequence in enumerate(outputs.sequences):
        sequence_scores = outputs.scores[:max_length]
        log_prob_sequence = 0.0
        
        for t, logits in enumerate(sequence_scores):
            token_id = sequence[len(input_ids[0]) + t].item()
            logits = logits.float()  # 确保计算的数值稳定性
            log_prob = torch.log_softmax(logits, dim=-1)[0, token_id].item()
            log_prob_sequence += log_prob
        
        log_probs.append(log_prob_sequence)
    
    if all(lp == float('-inf') for lp in log_probs):
        log_probs = [0.0 for _ in log_probs]  # 防止所有 log_probs 都为负无穷的情况
    
    return candidates, log_probs


def build_prompt(state, topic, hot_comment="暂无最新热门评论", mean_field='', related_cases_info=''):
    """
    构建优化后的 prompt，包括用户信息、主题、关键词以及相关案例。
    """
    prompt = f"当前讨论的主题是：{topic}"
    if hot_comment != "暂无最新热门评论":
        prompt += f"目前最热的评论可能是：{hot_comment}。"
    if mean_field != '':
        prompt += f"其他网友讨论的总体情况是：{mean_field}。"
    if related_cases_info != '':
        prompt += f"以下最相关的案例是：\n{related_cases_info}。"
    
    prompt_text = prompt + f"{state} 模仿该网友生成一条微博评论："
    return prompt_text


def calculate_log_probs(model, tokenizer, prompts, true_actions, max_length, device):
    """
    计算给定 prompts 和 true actions 的 log 概率。
    """
    prompt_tokens = tokenizer(prompts, truncation=True, padding=True, return_tensors="pt").to(device)
    true_action_tokens = \
    tokenizer(true_actions, max_length=max_length, truncation=True, padding=True, return_tensors="pt").to(device)[
        "input_ids"]
    
    with autocast("cuda"):
        outputs = model(**prompt_tokens, return_dict=True)
    logits = outputs.logits
    
    log_probs = F.log_softmax(logits, dim=-1)
    log_p_action_true = log_probs.gather(2, true_action_tokens.unsqueeze(-1)).squeeze(-1).sum(dim=-1)
    
    return log_p_action_true


def calculate_weighted_log_p_next_action(model, tokenizer, generated_mean_fields, probs_m, next_states, topics,
                                         next_actions_true, max_length, device):
    """
    计算加权的 log_p_next_action，仅对概率 > 0 的 prompts 进行批量推理。
    """
    all_prompts = []
    valid_indices = []
    
    for i, mf_set in enumerate(generated_mean_fields):
        for j, message in enumerate(mf_set):
            if probs_m[i][j].item() > 0:  # 只对概率 > 0 的进行处理
                combined_prompt = build_prompt(state=next_states[i], mean_field=message, topic=topics[i])
                all_prompts.append(combined_prompt)
                valid_indices.append((i, j))
    
    # 对所有有效 prompts 一次性进行批量处理
    prompt_tokens = tokenizer(all_prompts, truncation=True, padding=True, return_tensors="pt").to(device)
    with autocast("cuda"):
        outputs = model(**prompt_tokens, return_dict=True)
    logits = outputs.logits
    
    # 获取所有 true action 的 tokens
    all_true_tokens = []
    for next_action in next_actions_true:
        tokens = tokenizer(next_action, max_length=max_length, truncation=True, return_tensors="pt").to(device)[
            "input_ids"]
        all_true_tokens.append(tokens)
    
    all_true_tokens = torch.cat(all_true_tokens, dim=0)
    
    # 分别计算每个候选项的概率
    log_probs_combined = F.log_softmax(logits, dim=-1)
    weighted_log_p_next_action = [0.0] * len(generated_mean_fields)
    
    for idx, (i, j) in enumerate(valid_indices):
        next_action_true_tokens = all_true_tokens[i].unsqueeze(0)
        logits_slice = log_probs_combined[idx]
        log_p_next_action = logits_slice.gather(1, next_action_true_tokens).squeeze(-1).sum().item()
        weighted_log_p_next_action[i] += probs_m[i][j].item() * log_p_next_action
    
    return torch.tensor(weighted_log_p_next_action).to(device)


def calculate_loss_batch(
        model, tokenizer, states, topics, actions_true, next_states, next_actions_true, previous_mf_tokens, top_k,
        max_length
):
    """
    批量计算损失值。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = len(states)
    
    # 确保模型权重为 FP16 或混合精度
    model.to(torch.float16).to(device)
    
    # 计算当前 log_p_action_true
    batch_prompts = [
        build_prompt(state=states[i], mean_field=previous_mf_tokens[i], topic=topics[i])
        for i in range(batch_size)
    ]
    log_p_action_true = calculate_log_probs(model, tokenizer, batch_prompts, actions_true, max_length, device)
    
    # 准备 mean field 并生成候选项
    batch_mean_field = [
        f"当前的网友描述是：{states[i]}，对应评论内容是：{actions_true[i]}。重点结合过去网友们评论的总体情况（{previous_mf_tokens[i]}），更新并提取出影响后续网友评论的关键性总体信息，请用一句话概括："
        for i in range(batch_size)
    ]
    batch_mean_field_tokens = tokenizer(batch_mean_field, truncation=True, padding=True, return_tensors="pt").to(device)
    
    all_candidates = []
    all_probs_m = []
    
    for i in range(batch_size):
        candidates, log_probs = generate_candidates_with_log_probs(
            batch_mean_field_tokens["input_ids"][i].unsqueeze(0), top_k, model, tokenizer
        )
        probs_m = torch.softmax(torch.tensor(log_probs).float(), dim=-1)
        all_candidates.append(candidates)
        all_probs_m.append(probs_m)
    
    # 计算加权 log_p_next_action
    weighted_log_p_next_action = calculate_weighted_log_p_next_action(
        model, tokenizer, all_candidates, all_probs_m, next_states, topics, next_actions_true, max_length, device
    )
    
    # 计算损失
    losses = -(log_p_action_true + weighted_log_p_next_action)
    avg_loss = losses.mean()
    
    # 释放内存
    torch.cuda.empty_cache()
    
    return avg_loss, all_candidates


if __name__ == "__main__":
    set_random_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = "/home/mqr/code/llama_factory/language_model/Qwen2-7B-Instruct"  # 替换为你的模型路径
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    
    max_length = 256
    top_k = 3
    
    # 示例数据
    batch_states = [
        "作为一位来自北京 朝阳区，自我描述为热爱生活的网友，你的朋友数量为1024个，粉丝数量为2035个。",
        "作为一位来自上海 浦东新区，自我描述为科技爱好者的网友，你的朋友数量为578个，粉丝数量为1560个。",
        "作为一位来自深圳 南山区，自我描述为运动达人和美食博主的网友，你的朋友数量为812个，粉丝数量为2219个。"
    ]
    
    batch_topics = [
        "这是一个极具争议性的社会话题。",
        "最新的科技进展引发了广泛关注。",
        "关于健康生活方式的讨论持续升温。"
    ]
    
    batch_actions_true = [
        "支持楼主，转发扩散！",
        "这太可怕了，必须要举报！",
        "这些行为令人发指，强烈谴责！"
    ]
    
    batch_next_states = [
        "作为一位来自天津 南开区，自我描述为社会观察者的网友，你的朋友数量为435个，粉丝数量为1203个。",
        "作为一位来自杭州 西湖区，自我描述为文艺青年的网友，你的朋友数量为390个，粉丝数量为980个。",
        "作为一位来自成都 武侯区，自我描述为热衷公益事业的网友，你的朋友数量为620个，粉丝数量为1340个。"
    ]
    
    batch_next_actions_true = [
        "这件事情一定要曝光，让更多人知道！",
        "我们不能沉默，必须采取行动！",
        "团结一致，共同维护社会公正！"
    ]
    
    batch_previous_mf_tokens = [
        "讨论氛围非常热烈，许多网友都表达了强烈的不满。",
        "大多数网友认为应该采取更严厉的措施来遏制这种行为。",
        "大家普遍支持楼主，希望引起更广泛的关注。"
    ]
    
    loss, top_k_messages = calculate_loss_batch(
        model, tokenizer,
        batch_states, batch_topics, batch_actions_true, batch_next_states,
        batch_next_actions_true, batch_previous_mf_tokens, top_k, max_length
    )
    
    print("Batch Loss:", loss.item())
    print("Top-k Messages:", top_k_messages)
