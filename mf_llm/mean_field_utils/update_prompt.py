import torch
import openai
import torch.nn.functional as F
from collections import Counter
import time
# from transformers import AutoTokenizer, AutoModelForCausalLM
# MODEL_PATH = "/mnt/nasdata/qirui/language_model/"
# MODEL_NAME = "Qwen2-1.5B-Instruct"
# model_name = MODEL_PATH + MODEL_NAME
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# initial_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
# from collections import Counter
import os
import re
import re
import torch
import openai


def calculate_mean_field(
        topic,
        states,
        actions,
        pre_mean_field,
        model,
        device,
        alg,
        model_type,
        client=None,
        sampling_params=None,
        use_real_action=False,
        tokenizer=None,
        use_vllm=True,
        return_real_comment=False,
):
    """
    使用 LLM 或 GPT 计算指定索引的 mean field。
    Args:
        model: vLLM 模型实例或本地模型实例。（若 use_gpt=True，则可忽略本地模型）
        device: 设备。
        sampling_params: LLM 的生成参数（如果是vLLM或本地模型可以用到）。
        use_real_action: 是否使用真实评论还是生成的评论作为 a_prev。
        tokenizer: 本地模型的tokenizer。
        use_vllm: 是否使用 vLLM。
        use_gpt: 是否使用 OpenAI GPT。
    Returns:
        当前索引对应的 mean field 字符串。
    """

    state_action_prompt = ""

    end_prompt = \
        ("请你结合以上信息，总结网友的评论分布情况，方便后续网友快速了解已有讨论内容，对以下 6 个方面，根据重要性的顺序进行回答："
         "1. 立场分布：大部分网友们是持有反对意见，还是支持？"
         "2. 观点分布：网友们有哪些观点？请总结主要的观点内容"
         "3. 情绪分布：大部分网友是生气、兴奋、质疑、焦虑?是积极倾向、消极，还是中立？"
         "4. 行为分布：他们更倾向于转发，还是评论？"
         "5. 话题真实性：网友们如何评价该话题的真实性？网友们是相信，还是质疑？"
         "6. 发言意图：网友的评论主要是提问、发表个人观点，还是转播信息？"
         "任务：请使用 200 字左右的中文总结回答，并确保信息结构清晰、简洁明了。")
         # "请你基于以上信息，写一段约 200 字的总结。")
    

    for i, (s_prev, a_prev) in enumerate(zip(states, actions)):
        # state_action_prompt += f"第{i + 1}个网友的个人资料：{s_prev}\n该网友的评论：{a_prev}。\n"
        state_action_prompt += f"第{i + 1}个网友评论：{a_prev}\n"
    prompt = (
                 f"当前话题：{topic}\n"
                 f"过往网友评论总结：{pre_mean_field}。\n"
                 # f"以下是当前网友最新评论及其个人资料：\n{state_action_prompt}\n") + end_prompt
                 f"最新网友讨论情况：\n{state_action_prompt}\n") + end_prompt
    if return_real_comment == False:
        if "gpt" in model_type:
            try:
                response = client.chat.completions.create(
                    model=model_type,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,  # do_sample=False，相当于 temperature=0
                    max_tokens=200,  # max_new_tokens 对应 max_tokens
                    top_p=1.0,  # 不设置 top_p，使其与 greedy decoding 一致
                )
                generated_mean_field = response.choices[0].message.content.strip().split("Human:")[0].split("Assistant:")[0].replace("\n", "")
            except Exception as e:
                print(f"调用OpenAI API出错: {e}")
                generated_mean_field = "（OpenAI API 调用失败）"
            return generated_mean_field, torch.tensor(0.)
        if "DeepSeek" in model_type:
            messages = [
                {"role": "user", "content": "请直接输出最终答案，不要展示思考过程。\n" + prompt},
                {"role": "assistant", "content": "<think>\n</think>\n\n"}  # 引导跳过思考
            ]
            try:
                response = client.chat.completions.create(
                    model=model_type,
                    messages=messages,
                    temperature=0.1,  # do_sample=False，相当于 temperature=0
                    max_tokens=650,
                )
                generated_mean_field = re.sub(r'<think>.*?</think>\s*', '', response.choices[0].message.content, flags=re.DOTALL).strip().replace("\n", "").replace("**", "")
            except Exception as e:
                print(f"调用OpenAI API出错: {e}")
                generated_mean_field = "（OpenAI API 调用失败）"
            return generated_mean_field, torch.tensor(0.)
        # # ------------------ 如果使用 vLLM ------------------
        # elif use_vllm:
        #     result = model.generate(prompt, sampling_params)
        #     generated_mean_field = result[0].outputs[0].text.strip()
        else:
            # ------------------ 否则使用本地模型 ------------------
            generated_texts = model_generate(prompt, model, tokenizer,device)
            # m_log_probs = compute_loss(model, tokenizer, prompt, end_prompt, generated_texts, device)
            # return generated_texts, m_log_probs
            return generated_texts, torch.tensor([0.], device=device)
    else:
        return (f"以下是当前网友最新评论及其个人资料：\n{state_action_prompt}\n "), torch.tensor(0.)
    
    
    
def model_generate(prompt, model, tokenizer, device):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(device)["input_ids"]
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            # max_new_tokens=300,
            max_new_tokens=200,
            min_length=inputs.shape[1] + 10,
            do_sample=True,
            temperature=0.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2
        )

    
    generated_texts = \
    tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True).strip().split("Human:")[0].split(
        "Assistant:")[0].replace("\n", "")
    if generated_texts == "":
        return tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)

    return generated_texts
    
    

def compute_loss(model, tokenizer, prompt, end_prompt, generated_texts, device, beta=0.5):
    q_m_consition_logp = compute_log_prob_no_condition(model, tokenizer, [prompt] * 1, [generated_texts] * 1, device)
    q_m_logp = compute_log_prob_no_condition(model, tokenizer, [end_prompt] * 1, [generated_texts] * 1, device)
    with torch.no_grad():
        intial_m_logp = compute_log_prob_no_condition(initial_model.to(device), tokenizer, [end_prompt] * 1, [generated_texts] * 1,
                                                  device)
        tokens = tokenizer.tokenize(generated_texts)
        token_counts = Counter(tokens)  # 统计 Token 频率
        freq_penalty = sum((count - 1) for count in token_counts.values() if count > 1)
    total_mf_objective = q_m_consition_logp - q_m_logp - beta * (intial_m_logp -q_m_logp )- freq_penalty*0.002
    print(f"p(m|m,s,a)={q_m_consition_logp.item()}; p(m)={q_m_logp.item()}; r(m)={intial_m_logp.item()}; freq_penalty={freq_penalty*0.002}")

    return total_mf_objective


def compute_log_prob_no_condition(model, tokenizer, prompts, texts, device):
    inputs = [prompt + text for prompt, text in zip(prompts, texts)]
    tokenized_inputs = tokenizer(inputs, return_tensors="pt", truncation=True, padding=True).to(device)
    tokenized_prompts = tokenizer(prompts, return_tensors="pt", truncation=True, padding=True).to(device)

    input_ids = tokenized_inputs["input_ids"]
    prompt_ids = tokenized_prompts["input_ids"]
    attention_mask = tokenized_inputs["attention_mask"]
    prompt_lengths = [len(ids) for ids in prompt_ids]

    outputs = model(input_ids, attention_mask=attention_mask, return_output=True)
    logits = outputs.logits

    sequence_avg_probs = []
    for j in range(input_ids.shape[0]):
        log_prob_sum = torch.tensor(0.0, device=input_ids.device, dtype=torch.float32, requires_grad=True)
        valid_token_count = 0

        for step in range(prompt_lengths[j], input_ids.shape[1]):
            token_id = input_ids[j, step]
            log_prob_sum = log_prob_sum + F.log_softmax(logits[j, step], dim=-1)[token_id]
            valid_token_count += 1

        if valid_token_count > 0:
            # sequence_avg_prob = log_prob_sum
            sequence_avg_prob = log_prob_sum / valid_token_count
        else:
            sequence_avg_prob = torch.tensor(-100.0, device=input_ids.device, dtype=torch.float32, requires_grad=True)

        sequence_avg_probs.append(sequence_avg_prob)

    return torch.stack(sequence_avg_probs)  # Keeps gradients





def build_state(item):
    """Build a semantic state string by categorizing user information and predicting possible text content."""
    # Basic user attributes
    user_location = item.get('user_location', '未知地区')
    user_description = item.get('user_description', '未知')
    friends_count = item.get('friends_count', 0)
    followers_count = item.get('followers_count', 0)
    verified = item.get('verified', False)
    verified_type = item.get('verified_type', -1)
    gender = "男性" if item.get('gender') == 'm' else "女性"
    
    # Interaction data
    reposts_count = item.get('reposts_count', 0)
    comments_count = item.get('comments_count', 0)
    attitudes_count = item.get('attitudes_count', 0)
    
    # Categorize friends_count
    if friends_count < 10:
        friends_level = "非常少"
    elif 10 <= friends_count <= 30:
        friends_level = "较少"
    elif 31 <= friends_count <= 1000:
        friends_level = "适中"
    elif 1001 <= friends_count <= 3000:
        friends_level = "较多"
    else:
        friends_level = "非常多"
    
    # Categorize followers_count as influence level
    if followers_count < 100:
        influence_level = "非常小"
    elif 101 <= followers_count <= 500:
        influence_level = "较小"
    elif 501 <= followers_count <= 1000:
        influence_level = "适中"
    elif 1001 <= followers_count <= 10000:
        influence_level = "较大"
    else:
        influence_level = "非常大"
    
    # Determine activity level
    total_interactions = reposts_count + comments_count + attitudes_count
    if total_interactions < 10:
        activity_level = "不活跃"
    elif 10 <= total_interactions <= 100:
        activity_level = "较活跃"
    else:
        activity_level = "非常活跃"
    
    # Verified status description
    if verified:
        verified_status = f"认证用户（类型 {verified_type}）"
    else:
        verified_status = "非认证用户"
    
    # Build state description
    state = (
        f"来自{user_location}，描述为{user_description}的{gender}网友，"
        f"朋友数量{friends_level}，影响力{influence_level}，"
        f"动态互动情况为{activity_level}，{verified_status}。"
    
    )
    return state


def build_prompt(state, topic, hot_comment, mean_field, related_cases_info, alg="none", model_type=None):
    """
    构建优化后的 prompt，包括用户信息、主题、关键词以及相关案例。
    Args:
        state: 当前用户的状态信息。
        topic: 当前讨论的主题。
        mean_field: 当前的 mean field。
        related_cases_info: 最相关案例的信息。
    Returns:
        包含用户信息、主题、关键词和案例的 prompt。
    """

    prompt = f"当前讨论的话题是：{topic}"
    
    if hot_comment != "暂无最新热门评论":
        prompt += f"{hot_comment}"
    if mean_field != '':
        prompt += f"重点参考已有的网友们评论情况：{mean_field}。\n"
    if related_cases_info != '':
        prompt += f"以下最相关的案例是：\n{related_cases_info}。\n"
    # 用户信息
    prompt += f"网友资料：{state}\n"
    

    if "gpt" in model_type or "DeepSeek" in model_type:
        prompt += (
            "根据其他网友的评论情况，请推测该网友可能的情绪、观点和立场，模拟该网友进行转发微博或评论，要求直接输出模拟的内容。")
    else:
        if alg == "state_sft":
            prompt += (
                "根据上述信息，模仿该网友进行转发微博或评论："
            )
        else:
            prompt += (
                "根据其他网友的评论情况，请推测该网友可能的情绪、观点和立场，模拟该网友进行转发微博或评论：")
            # "根据上述信息，模仿该网友进行转发微博或评论：")

    return prompt


