from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def generate_candidates_with_log_probs(input_ids, top_k, model, tokenizer, max_length=50, top_p=0.95, temperature=0.7):
    """
    根据输入 prompt 生成 top_k 个候选序列，并计算其 log 概率。

    Args:
        input_ids (torch.Tensor): 输入的 token 化 prompt。
        top_k (int): 生成的候选序列数量。
        model: 预训练语言模型。
        tokenizer: 用于解码的 tokenizer。
        max_length (int): 每个生成序列的最大长度。
        top_p (float): Nucleus Sampling 的概率阈值。
        temperature (float): 控制采样随机性的温度参数。

    Returns:
        candidates (list of str): 生成的候选序列。
        log_probs (list of float): 每个候选序列的累积 log 概率。
    """
    # 使用模型生成候选序列，仅限制生成内容的长度
    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_length,  # 仅限制生成内容的长度
        num_return_sequences=top_k,
        do_sample=True,
        top_p=top_p,
        temperature=temperature,
        output_scores=True,  # 启用 scores 输出
        return_dict_in_generate=True  # 返回生成的字典
    )

    # 解码生成的候选序列，仅保留生成内容
    candidates = [
        tokenizer.decode(output[len(input_ids[0]):], skip_special_tokens=True)
        for output in outputs.sequences
    ]

    # 计算每个候选序列的累积 log 概率，仅包含生成部分
    log_probs = []
    for sequence_idx, sequence in enumerate(outputs.sequences):
        sequence_scores = outputs.scores[:max_length]  # 只处理生成部分的 logits
        log_prob_sequence = 0.0

        for t, logits in enumerate(sequence_scores):
            token_id = sequence[len(input_ids[0]) + t].item()  # 跳过 input_ids 部分
            log_prob = torch.log_softmax(logits, dim=-1)[0, token_id].item()
            log_prob_sequence += log_prob

        log_probs.append(log_prob_sequence)

    return candidates, log_probs

# 示例调用
if __name__ == "__main__":
    prompt = "The future of AI is"
    top_k = 3
    # 示例用法
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = "/home/mqr/code/llama_factory/language_model/Qwen2-1.5B-Instruct"  # 替换为你的模型路径
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    candidates, log_probs = generate_candidates_with_log_probs(prompt, top_k, model, tokenizer)

    print("Generated Candidates with Log Probabilities:")
    for idx, (candidate, log_prob) in enumerate(zip(candidates, log_probs)):
        print(f"{idx+1}: {candidate} (log_p: {log_prob:.4f})")
