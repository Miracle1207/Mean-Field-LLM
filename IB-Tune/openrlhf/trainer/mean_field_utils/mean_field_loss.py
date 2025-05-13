
from openrlhf.models import GPTLMLoss
import torch
from collections import Counter
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "" # todo
MODEL_NAME = "" # todo: you need load a pretrain LLM as initial model; For example, Qwen2-1.5B
model_name = MODEL_PATH + MODEL_NAME
initial_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
# from collections import Counter

def top_p_filtering(log_probs, top_p=0.9):
    sorted_indices = torch.argsort(log_probs, descending=True)
    cumulative_probs = torch.cumsum(torch.exp(log_probs[sorted_indices]), dim=-1)
    indices_to_remove = cumulative_probs > top_p
    log_probs[sorted_indices[indices_to_remove]] = float('-inf')
    return log_probs

def build_prompt(state, topic, hot_comment="暂无最新热门评论", mean_field= '', related_cases_info= ''):
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
    # 完整的 prompt 包括用户信息

    prompt = f"当前讨论的话题是：{topic}"
    if hot_comment != "暂无最新热门评论":
        prompt += f"目前最热的评论是：{hot_comment}。"
    if mean_field != '':
        prompt += f"目前网友们评论的总体情况为：{mean_field}。\n"
    if related_cases_info != '':
        prompt += f"以下最相关的案例是：\n{related_cases_info}。\n"
    # 用户信息
    prompt += f"网友资料：{state}\n"

    # 明确任务指令
    prompt += (
        "根据其他网友的评论情况，请推测该网友可能的情绪、观点和立场，模拟该网友进行转发微博或评论："
    )
    return prompt


def calculate_mean_field(
        topic,
        states,
        actions,
        pre_mean_field,
        model,
        device,
        number_m=1,
        tokenizer=None,
        compute_mf_loss=True
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

    for i, (s_prev, a_prev) in enumerate(zip(states, actions)):
        state_action_prompt += f"第{i + 1}个网友的个人资料：{s_prev}\n该网友的评论：{a_prev}。\n"
    prompt = (
        f"当前话题：{topic}\n"
        f"基于过往网友评论总结：{pre_mean_field}。\n"
        f"以下是当前网友最新评论及其个人资料：\n{state_action_prompt}\n" ) + end_prompt
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(device)["input_ids"]

    # A = time.time()
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=200,
            do_sample=True,  # 启用采样
            temperature=0.1,  # 设置温度
            top_k=30,  # 设置 Top-K
            top_p=0.95,  # 设置 Top-P
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    generated_texts = tokenizer.decode(outputs[0][inputs.shape[1]:],skip_special_tokens=True).strip().split("Human:")[0].replace("\n", "")
    
    if compute_mf_loss == True:
        m_log_probs = compute_loss(model, tokenizer, prompt, end_prompt, generated_texts, device)
    else:
        m_log_probs = torch.zeros(number_m)
    
    best_mean_field = generated_texts
    return best_mean_field, m_log_probs


def compute_loss(model, tokenizer, prompt, end_prompt, generated_texts, device, beta=1):
    q_m_consition_logp = compute_log_prob_no_condition(model, tokenizer, [prompt] * 1, [generated_texts] * 1, device)
    intial_m_logp = compute_log_prob_no_condition(initial_model.to(device), tokenizer, [end_prompt] * 1,
                                                  [generated_texts] * 1,
                                                  device)
    tokens = tokenizer.tokenize(generated_texts)
    token_counts = Counter(tokens)  # 统计 Token 频率
    freq_penalty = sum((count - 1) for count in token_counts.values() if count > 1)
    total_mf_objective = q_m_consition_logp - beta * (intial_m_logp)- freq_penalty*0.002
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

def calculate_loss_batch(
        alg,
        model, tokenizer,
        states, actions_true,
        previous_mf,
        use_true_action=False,
        mf_model=None,
        device="cuda"
):
    if alg == "state":
        action_loss, mean_field = calculate_sa_loss_batch(
                model=model,
                tokenizer=tokenizer,
                states=states,
                actions_true=actions_true,
                previous_mf=previous_mf,
                use_true_action=use_true_action,
                device=device)
        mf_loss = 0
        policy_loss = 0
    elif alg == "mf":
        action_loss,policy_loss, mf_loss, mean_field = calculate_mf_loss_batch(
                model=model,
                tokenizer=tokenizer,
                states=states,
                actions_true=actions_true,
                previous_mf=previous_mf,
                use_true_action=use_true_action,
                device=device)

    elif alg == "policy_on_mf":
        action_loss,policy_loss, mf_loss, mean_field = calculate_mf_loss_batch(
            model=model,
            tokenizer=tokenizer,
            states=states,
            actions_true=actions_true,
            previous_mf=previous_mf,
            use_true_action=use_true_action,
            mf_model=mf_model,
            compute_mf_loss=False,
            device=device)
    else:
        raise ValueError(f"Unsupported algorithm: {alg}")
    
    return action_loss, policy_loss, mf_loss, mean_field

def calculate_mf_loss_batch(
        model, tokenizer,
        states, actions_true,
        previous_mf,
        use_true_action=True,
        compute_mf_loss=True,
        mf_model=None,
        max_new_tokens=80,
        device="cuda"
):
    """
    Calculate loss by generating the current mean field, then using it with the next state
    to predict the next action.

    Args:
        model, tokenizer: Pretrained model and tokenizer.
        states: [B] Current states (text descriptions).
        actions_true: [B] Current true actions (text descriptions).
        previous_mf: [B] Previous mean field (text descriptions).
        max_new_tokens: Maximum length for mean field generation.
        device: Device for computation.
    Returns:
        total_loss: Combined loss for the batch.
        current_mfs: Generated current mean fields for each sample in the batch.
    """
    B = len(states)
    model = model.to(device)
    topic = states[0]['topic']
    states_info = [states[i]['state'] for i in range(B)]
    
    # todo: 1. log prob(a* | mt, st)
    action_prompts = [
        build_prompt(state=states[i]['state'],topic=states[i]['topic'], mean_field=previous_mf)
        for i in range(B)
    ]
    action_prompts_wo_mf = [
        build_prompt(state=states[i]['state'], topic=states[i]['topic'])
        for i in range(B)
    ]
    
    if use_true_action:
        generated_comments = actions_true
    else:
        try:
            inputs = tokenizer(
                action_prompts,
                return_tensors="pt",
                truncation=True,
                padding=True
            ).to(device)
            input_tokens = inputs["input_ids"]
            input_len = inputs['attention_mask'].shape[1]  # 计算有效长度   tokenizer.padding_side = "left"
            with torch.no_grad():
                outputs = model.generate(
                    input_tokens,
                    max_new_tokens=30,
                    do_sample=True,
                    temperature=0.5,
                    top_k=50,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.2
                )
            outputs_list = outputs.tolist()

            generated_comments = []
            for i, output in enumerate(outputs_list):
                new_tokens = output[input_len:]  # 只取生成的新 token
                generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip().split('\n')[0]
                generated_comments.append(generated_text)
        except Exception as e:
            print(f"调用本地模型出错: {e}")
    
    if compute_mf_loss == False:
        action_loss = calculate_log_probs(
            model=model,
            # model=initial_model.to(device),
            tokenizer=tokenizer,
            prompts=action_prompts,  # [B] list[str]
            true_actions=actions_true,  # [B] list[str]
            max_length=500,
            device=device
        )
        initial_action_loss = calculate_log_probs(
            # model=model,
            model=initial_model.to(device),
            tokenizer=tokenizer,
            prompts=action_prompts_wo_mf,  # [B] list[str]
            true_actions=actions_true,  # [B] list[str]
            max_length=200,
            device=device
        )
        policy_loss =  action_loss - initial_action_loss  # -logp(a|m,s) -(-logp(a|s))
    else:
        action_loss = calculate_log_probs(
        # model=model,
        model=initial_model.to(device),
        tokenizer=tokenizer,
        prompts=action_prompts,  # [B] list[str]
        true_actions=actions_true,  # [B] list[str]
        max_length=500,
        device=device
        )
        policy_loss = action_loss

    # todo: step1. generate m_t
    mf_source = model if mf_model is None else mf_model
    mean_field, kl_loss = calculate_mean_field(
        topic,
        states_info,
        actions_true,
        previous_mf,
        mf_source,
        device=device,
        tokenizer=tokenizer,
        compute_mf_loss=compute_mf_loss,
    )

    return action_loss, policy_loss, kl_loss, mean_field


def calculate_sa_loss_batch(
        model, tokenizer,
        states, actions_true,
        previous_mf,
        use_true_action=False,
        max_new_tokens=80,
        device="cuda"
):
    """
    Calculate loss by generating the current mean field, then using it with the next state
    to predict the next action.

    Args:
        model, tokenizer: Pretrained model and tokenizer.
        states: [B] Current states (text descriptions).
        actions_true: [B] Current true actions (text descriptions).
        previous_mf: [B] Previous mean field (text descriptions).
        max_new_tokens: Maximum length for mean field generation.
        device: Device for computation.
    Returns:
        total_loss: Combined loss for the batch.
        current_mfs: Generated current mean fields for each sample in the batch.
    """
    B = len(states)
    model = model.to(device)
    
    # todo: 1. log prob(a* | m, s)
    action_prompts = [
        build_prompt(state=states[i]['state'], topic=states[i]['topic'])
        for i in range(B)
    ]
    
    if use_true_action:
        generated_comments = actions_true
    else:
        try:
            inputs = tokenizer(
                action_prompts,
                return_tensors="pt",
                truncation=True,
                padding=True
            ).to(device)
            input_tokens = inputs["input_ids"]
            input_len = inputs['attention_mask'].shape[1]  # 计算有效长度   tokenizer.padding_side = "left"
            with torch.no_grad():
                outputs = model.generate(
                    input_tokens,
                    max_new_tokens=30,
                    do_sample=True,
                    temperature=0.5,
                    top_k=50,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.2
                )
            outputs_list = outputs.tolist()
            
            generated_comments = []
            for i, output in enumerate(outputs_list):
                new_tokens = output[input_len:]  # 只取生成的新 token
                generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip().split('\n')[0]
                generated_comments.append(generated_text)
        except Exception as e:
            print(f"调用本地模型出错: {e}")
    
    action_loss = calculate_log_probs(
        model=model,
        tokenizer=tokenizer,
        prompts=action_prompts,  # [B] list[str]
        true_actions=actions_true,  # [B] list[str]
        max_length=400,
        device=device
    )
    
    return action_loss, ""


def calculate_log_probs(model, tokenizer, prompts, true_actions, max_length, device, packing_samples=False,
                        pretrain_mode=False, ring_attn_group=None):
    """
    Calculate the Cross-Entropy Loss for a given model, prompts, and true actions.

    Args:
        model: The model to use.
        tokenizer: The tokenizer corresponding to the model.
        prompts (list of str): List of input prompts.
        true_actions (list of str): List of true action strings for each prompt.
        max_length (int): Maximum sequence length for the model.
        device (str): Device to use (e.g., "cpu" or "cuda").
        packing_samples (bool): Whether the inputs are packed samples.
        pretrain_mode (bool): Whether the model is in pretraining mode.
        ring_attn_group: Optional ring attention group for distributed training.

    Returns:
        float: Average Cross-Entropy Loss across all prompts.
    """
    # Initialize GPTLMLoss with optional ring_attn_group
    loss_fn = GPTLMLoss(ring_attn_group=ring_attn_group)
    
    processed_true_actions = [
        action if action.strip() != "" else "转发微博" for action in true_actions
    ]
    
    # Combine each prompt with its corresponding true_action
    combined_texts = [f"{prompt}{action}" for prompt, action in zip(prompts, processed_true_actions)]
    
    # Tokenize prompts
    tokenized_prompts = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    ).to(device)
    
    # Tokenize combined prompts and true actions
    tokenized_combined = tokenizer(
        combined_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    ).to(device)
    
    input_ids = tokenized_combined["input_ids"]
    labels = input_ids.clone()
    prompts_ids = tokenized_prompts["input_ids"]
    attention_mask = tokenized_combined["attention_mask"]
    
    # For each sample in the batch, set the labels corresponding to the prompt tokens to IGNORE_INDEX
    # This ensures that the loss is only computed for the true_actions
    for i in range(labels.size(0)):
        prompt_length = (prompts_ids[i] != tokenizer.pad_token_id).sum().item()
        labels[i, :prompt_length] = loss_fn.IGNORE_INDEX
    
    # Forward pass with model
    if ring_attn_group is None:
        outputs = model(input_ids, attention_mask=attention_mask, return_output=True)
    else:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_output=True,
            ring_attn_group=ring_attn_group,
        )
    
    # Extract logits from model outputs
    logits = outputs.logits
    
    # Compute the loss
    loss = loss_fn(logits, labels)

    if torch.isnan(loss):
        print("Warning: Loss is NaN")
        return torch.tensor(10., device=device, dtype=torch.bfloat16)
    
    return loss

