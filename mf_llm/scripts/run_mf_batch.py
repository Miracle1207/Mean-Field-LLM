import os
import json
import numpy as np
import random
import torch
from datetime import datetime
from mf_llm.mean_field_utils import calculate_log_probs, calculate_mean_field, build_state, build_prompt

seed_num = 46
# 设置随机种子与设备
random.seed(seed_num)
np.random.seed(seed_num)
torch.manual_seed(seed_num)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
from transformers import AutoTokenizer, AutoModelForCausalLM

import pandas as pd

import asyncio
import aiohttp
import re

async def async_call_single(i, prompt, model_name, api_base, temperature,top_p,max_tokens):
    url = f"{api_base}/chat/completions"
    headers = {"Authorization": f"Bearer EMPTY"}
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": "要求思考过程简短。" + prompt},
            {"role": "assistant", "content": "<think>\n</think>\n\n"}
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as resp:
                data = await resp.json()
                content = data['choices'][0]['message']['content']
                cleaned = re.sub(r'<think>.*?</think>\s*', '', content, flags=re.DOTALL).strip().split('\n')[0]
                return i, cleaned
    except Exception as e:
        return i, "（OpenAI API 调用失败）"

async def async_generate_with_vllm(batch_prompts,temperature,top_p,max_tokens, model_name="deepseek-chat", api_base="http://172.18.129.58:23369/v1"):
    tasks = [async_call_single(i, prompt, model_name, api_base, temperature, top_p, max_tokens) for i, prompt in enumerate(batch_prompts)]
    results = await asyncio.gather(*tasks)
    results.sort()  # 保证按顺序返回
    return [r[1] for r in results]


def run_simulation(tweets, comment_n, tokenizer, a_model,mf_model,batch_size=8, alg="RAG",model_type='1.5B', generate_true=True, simulation_start=2):
    """
    运行模拟，针对给定的 tweets 和 topic。
    Args:
        tweets: 微博评论列表。
        comment_n: 需要评估的评论数量。
        batch_size: 每个模拟步骤处理的代理数量。
        alg: 算法类型，"RAG" 使用案例搜索，"None" 不使用案例, "gpt" 使用OpenAI。
        generate_true: 是否生成评论。
        simulation_start: 模拟起点，在此之前使用真实数据计算，之后使用生成数据。
    Returns:
        包含评估结果的 DataFrame。
    """
    topic = tweets[0]['text']  # 讨论主题
    sorted_tweets = sorted(tweets[1:], key=lambda x: x.get('t', 0))  # 排除第一个主题 tweet
    
    comment_n = min(len(sorted_tweets), comment_n)
    sampled_indices = list(range(comment_n))  # 选取前 comment_n 个索引
    print(f"agent num:{len(sorted_tweets)}")
    agents = [
        {
            'uid': sorted_tweets[idx]['uid'],
            'text': sorted_tweets[idx]['text'] if sorted_tweets[idx]['text'] else "转发微博" + sorted_tweets[idx]['original_text'],
            'user_description': sorted_tweets[idx].get('user_description', "未知"),
            'friends_count': sorted_tweets[idx].get('friends_count', 0),
            'followers_count': sorted_tweets[idx].get('followers_count', 0),
            'hot_count': sorted_tweets[idx].get('reposts_count', 0)
                         + sorted_tweets[idx].get('attitudes_count', 0)
                         + sorted_tweets[idx].get('comments_count', 0),
            'fans_count': sorted_tweets[idx].get('followers_count', 0)
                         + sorted_tweets[idx].get('bi_followers_count', 0),
            'original_index': idx  # 在排序中的索引
        } for idx in sampled_indices
    ]
    
    results = []
    
    total_steps = (len(agents) + batch_size - 1) // batch_size  # 计算总步数
    mean_field = ['']
    mf_loss = torch.tensor([0.], device=device)

    if "gpt" in model_type:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), )
    elif "ds" in model_type:
        from openai import OpenAI
        client = OpenAI(api_key="sk-XXX", base_url="https://api.deepseek.com")
    elif "DeepSeek" in model_type:
        from openai import OpenAI
        client = OpenAI(
            api_key="EMPTY",
            base_url="http://172.18.129.58:23369/v1",
        )
    else:
        client = None
        
    for step in range(total_steps):
        # 确定当前批次的代理
        start_idx = step * batch_size
        end_idx = min(start_idx + batch_size, len(agents))
        batch_agents_list = agents[start_idx:end_idx]
        batch_agents = pd.DataFrame(batch_agents_list)
        
        latest_hot_comment_texts = ['暂无最新热门评论'] * batch_size
        # 批量计算最新热门评论（若包含 hot 逻辑）
        if 'hot' in alg:
            relevant_results = results[max(len(results) - 30, 0):]
            top_hot_comments = sorted(
                relevant_results,
                key=lambda x: x.get('fans_count', 0),
                reverse=True
            )[:5]
    
            if top_hot_comments:
                all_hot_comment = "当前最热门的评论如下："
                is_before_simulation = start_idx < simulation_start
                if is_before_simulation:
                    label = 'real_comment'
                else:
                    label = 'generated_comment'
                for i, comment in enumerate(top_hot_comments):
                    all_hot_comment += f"第{i+1}位网友评论：{comment[label]}。\n"
                latest_hot_comment_texts = [all_hot_comment] * batch_size
            else:
                latest_hot_comment_texts = ['暂无最新热门评论'] * batch_size
        elif 'pre' in alg:
            relevant_results = results[max(len(results) - 5, 0):]
            if relevant_results:
                all_hot_comment = "最新的网友评论如下："
                is_before_simulation = start_idx < simulation_start
                if is_before_simulation:
                    label = 'real_comment'
                else:
                    label = 'generated_comment'
                for i, comment in enumerate(relevant_results):
                    all_hot_comment += f"第{i+1}位网友评论：{comment[label]}。\n"
                latest_hot_comment_texts = [all_hot_comment] * batch_size
            else:
                latest_hot_comment_texts = ['暂无最新热门评论'] * batch_size

        # 批量构建状态
        batch_states = [build_state(agent) for agent in batch_agents_list]

        related_cases_infos = [""] * batch_size
   
        
        print(f"Batch {step+1} mean field: {mean_field}")
        
        batch_prompts = [
            build_prompt(state, topic, latest_hot_comment_text, mean_field, related_cases_info, alg=alg, model_type=model_type)
            for state, latest_hot_comment_text, related_cases_info in zip(
                batch_states, latest_hot_comment_texts, related_cases_infos
            )
        ]

        batch_prompts_wo_mf = [
            build_prompt(state, topic, [''] * batch_size, "", related_cases_info, alg=alg, model_type=model_type)
            for state, latest_hot_comment_text, related_cases_info in zip(
                batch_states, latest_hot_comment_texts, related_cases_infos
            )
        ]
        
        # 批量生成评论
        
        batch_true_actions = list(batch_agents['text'])

        # generated_comments = ["转发微博" if item == "" else item for item in batch_true_actions]
        generated_comments = batch_true_actions
        if generate_true:
            is_before_simulations = start_idx + batch_size < simulation_start
            if not is_before_simulations:
                print("========= simulation ========")
                if "gpt" in model_type:
                    try:
                        for i, prompt in enumerate(batch_prompts):
                            response = client.chat.completions.create(
                                model=a_model,
                                messages=[{"role": "user", "content": prompt}],
                                temperature=0.8,  # 与 model.generate 保持一致
                                max_tokens=50,  # max_new_tokens 对应 max_tokens
                                top_p=0.85,  # 与 top_p 保持一致
                            )

                            generated_comments[i] = response.choices[0].message.content.split('\n')[0]
                        print(f"Batch {step + 1} generated_comment: {generated_comments}")
                    except Exception as e:
                        print(f"调用OpenAI API出错: {e}")
                        for i in range(batch_size):
                            generated_comments[i] = "（OpenAI API 调用失败）"
                if "DeepSeek" in model_type:  # local
                    try:
                        generated_comments = asyncio.run(
                            async_generate_with_vllm(
                                batch_prompts=batch_prompts,
                                model_name=a_model,
                                temperature=0.8,
                                top_p=0.85,
                                max_tokens=500
                            )
                        )
                        # print(f"Batch {step + 1} generated_comment: {generated_comments}")
                    except Exception as e:
                        print(f"调用 vLLM 异步接口出错: {e}")
                        generated_comments = ["（OpenAI API 调用失败）"] * batch_size

                elif "ds" in model_type:   # web
                    try:
                        for i, prompt in enumerate(batch_prompts):
  
                            response = client.chat.completions.create(
                                model="deepseek-chat",
                                messages=[
                                    {"role": "system", "content": "You are a helpful assistant"},
                                    {"role": "user", "content": prompt},
                                ],
                                temperature=0.8,  # 与 model.generate 保持一致
                                max_tokens=50,  # max_new_tokens 对应 max_tokens
                                top_p=0.85,  # 与 top_p 保持一致
                                stream=False,

                            )

                            generated_comments[i] = response.choices[0].message.content.split('\n')
                        print(f"Batch {step + 1} generated_comment: {generated_comments}")
                    except Exception as e:
                        print(f"调用OpenAI API出错: {e}")
                        for i in range(batch_size):
                            generated_comments[i] = "（OpenAI API 调用失败）"

                
                else:
                    generated_comments = generate_batch_comments(batch_prompts, tokenizer, a_model, device)
    
        else:
            print(f"Batch {step + 1} -- True comment = {batch_true_actions}")

        if alg == "mf_w_inter":
            # ## Intervention
            if step == 4:
                generated_comments[1] = "真的假的？"
                generated_comments[3] = "求真相"
                generated_comments[5] = "我不相信，奥运会又不是闹着玩的？"
                generated_comments[7] = "等待真相，不要传谣！请大家理智看待！！"

        elif alg == "mf_w_key":
            '''
            simulation with key notes:
            '''
            if step == 4:
                generated_comments[-1] = "真的假的？"
            if step == 5:
                generated_comments[1] = batch_true_actions[1]
                generated_comments[3] = batch_true_actions[3]

        print(f"Batch {step + 1} ---------------------------------------------")
        print('\n'.join(generated_comments))

        
        if "gpt" not in model_type and "ds" not in model_type and "DeepSeek" not in model_type:
            try:
                # 假设 calculate_log_probs 支持批量计算
                batch_log_probs = calculate_log_probs(
                    model=a_model,
                    tokenizer=tokenizer,
                    prompts=batch_prompts,
                    true_actions=batch_true_actions,
                    max_length=600,
                    device=device
                )

                print(f"Batch {step + 1} -- log p = {batch_log_probs}")
            except Exception as e:
                print(f"计算 log_prob 出错: {e}")
        else:
            batch_log_probs=10
            batch_log_probs_wo_mf=10


        for i, agent in enumerate(batch_agents_list):
            results.append({
                'step': step + 1,  # 当前步骤编号（批次编号）
                'agent_index': start_idx + i + 1,  # 代理的全局索引
                'topic': topic,
                'state': batch_states[i],
                'generated_comment': generated_comments[i],
                'real_comment': agent['text'],
                'log_prob': batch_log_probs,
                'mf_loss': mf_loss.item(),
                'mean_field': mean_field,
                'hot_count': agent['hot_count']
            })

        
        if "mf" in alg:
            mean_field, mf_loss = calculate_mean_field(
                topic,
                batch_states,
                generated_comments,
                mean_field,
                mf_model,
                device=device,
                alg=alg,
                model_type=model_type,
                client=client,
                # sampling_params=mf_sampling_params,
                use_real_action=False,
                tokenizer=tokenizer,
                use_vllm=use_vllm
            )
        
            
    
    return pd.DataFrame(results)


import torch


def generate_batch_comments(batch_prompts, tokenizer, a_model, device, max_new_tokens=50):
    """
    使用本地 Hugging Face 模型批量生成文本。

    参数:
        batch_prompts (List[str]): 输入提示列表
        tokenizer: Hugging Face tokenizer 对象
        a_model: Hugging Face 模型对象
        device: 模型使用的设备 (如 'cuda' 或 'cpu')
        max_new_tokens (int): 每个提示最大生成的新 token 数

    返回:
        List[str]: 生成的文本列表
    """
    try:
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            truncation=True,
            padding=True
        ).to(device)
        
        input_tokens = inputs["input_ids"]
        input_len = inputs['attention_mask'].shape[1]
        
        with torch.no_grad():
            outputs = a_model.generate(
                input_tokens,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_k=50,
                top_p=0.85,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2
            )
        
        outputs_list = outputs.tolist()
        generated_comments = []
        
        for i, output in enumerate(outputs_list):
            new_tokens = output[input_len:]  # 只取生成的新 token
            generated_text = \
            tokenizer.decode(new_tokens, skip_special_tokens=True).strip().replace("\u200b", "").split('\n')[0]
            generated_comments.append(generated_text)
        
        return generated_comments
    
    except Exception as e:
        print(f"调用本地模型出错: {e}")
        return []


def evaluate(test_dir, test_file_name, comment_n, alg, model_type, tokenizer, a_model,mf_model,batch_size, generate_true=True):
    """
    对测试目录下的文件进行评估。
    Args:
        test_dir: 测试数据所在目录。
        comment_n: 每个文件中需要评估的评论数量。
        generate_true: 是否生成评论。
    Returns:
        汇总的评估结果。
    """
    selected_files = [test_dir + '/'+test_file_name]
    all_results = pd.DataFrame()
    for file_i, file_path in enumerate(selected_files):
        print("-" * 50)
        print(f"TEST FILE {file_i}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        file_results = run_simulation(data, comment_n,tokenizer, a_model,mf_model, batch_size=batch_size, alg=alg,model_type=model_type, generate_true=generate_true, simulation_start=simulation_start)
        print(f"average log p = {file_results['log_prob'].mean()}, average mean field loss = {file_results['mf_loss'].mean()}")
        file_results['file'] = file_path
        all_results = pd.concat([all_results, file_results], ignore_index=True)
        
    return all_results

def runner(alg, model_type, test_name, comment_n, a_MODEL_NAME, mf_MODEL_NAME, generate_true=True):
    a_model_name = MODEL_PATH + a_MODEL_NAME
    mf_model_name = MODEL_PATH + mf_MODEL_NAME
    print(a_model_name)
    print(mf_model_name)
    test_dir = '../data/rumdect/Weibo/test'

    if "gpt" in model_type:
        # a_model = "gpt-4o-mini"
        a_model = model_type
        tokenizer = None
        
        mf_model= model_type
    elif "ds" in model_type:
        a_model = "deepseek-chat"
        tokenizer = None
        mf_model = None
    elif "DeepSeek" in model_type:
        a_model = model_type
        tokenizer = None
        mf_model = model_type

    else:
        if "mf" in alg:
            mf_model = AutoModelForCausalLM.from_pretrained(mf_model_name, torch_dtype=torch.float16).to(device)
            if a_MODEL_NAME != mf_MODEL_NAME:
                a_model = AutoModelForCausalLM.from_pretrained(a_model_name, torch_dtype=torch.float16).to(device)
            else:
                a_model = mf_model
        else:
            a_model = AutoModelForCausalLM.from_pretrained(a_model_name, torch_dtype=torch.float16).to(device)
            mf_model = None
        tokenizer = AutoTokenizer.from_pretrained(a_model_name)
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token

    evaluation_results = evaluate(test_dir, test_name,  comment_n, alg, model_type, tokenizer, a_model, mf_model,batch_size, generate_true=generate_true)
    
    return evaluation_results

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comment_n", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--simulation_start", type=int, default=100)
    parser.add_argument("--alg", type=str, default='mf')
    parser.add_argument("--model", type=str, default='1.5B') # "DeepSeek-R1-Distill-Qwen-32B"
    parser.add_argument("--task", type=str, default='')
    parser.add_argument("--file_name", type=str, default='retirement')
    args = parser.parse_args()
    return args

def find_file_info(file_name):
    for file_class, topics in data.items():  # Iterate over categories and their topic dictionaries
        if file_name in topics:  # Check if file_name exists in the current category
            return file_class, topics[file_name]  # Return category and filename
    return None, None  # Return None if not found


if __name__ == "__main__":
    args = parse_args()
    comment_n = args.comment_n
    generate_true = True
    ALG = args.alg  #  'hot', 'mf', 'gpt' 等
    simulation_start = args.simulation_start
    use_vllm = False
    MODEL_PATH = ""  # todo: fill your LLM path
    file_name = args.file_name
    model_type = args.model
    batch_size = args.batch_size
    
    if ALG == 'state' or ALG == 'hot' or ALG == 'pre':
        a_MODEL_NAME = f"Qwen2-{model_type}-Instruct"
        mf_MODEL_NAME = f"Qwen2-{model_type}-Instruct"
    elif ALG == 'mf' or ALG == 'mf_w_key' or ALG == 'mf_w_inter':
        if model_type == "1.5B":
            a_MODEL_NAME = "qwen2-1.5B-policy-0324-prompt"  # or "Qwen2-1.5B-Instruct"
            mf_MODEL_NAME = "qwen2-1.5B-mf-0319"    # or "Qwen2-1.5B-Instruct"
        elif model_type == "7B":
            a_MODEL_NAME = f"Qwen2-{model_type}-Instruct"
            mf_MODEL_NAME = f"Qwen2-{model_type}-Instruct"
        elif "gpt" in model_type:
            a_MODEL_NAME = model_type
            mf_MODEL_NAME = model_type
        elif "DeepSeek" in model_type:
            a_MODEL_NAME = model_type
            mf_MODEL_NAME = model_type
            
    elif ALG == 'mf_wo_sfta':
        a_MODEL_NAME = "Qwen2-1.5B-Instruct"
        mf_MODEL_NAME = "qwen2-1.5B-mf-0319"
    elif ALG == 'mf_wo_sft_mf':
        a_MODEL_NAME = "qwen2-1.5B-policy-0324-prompt"
        mf_MODEL_NAME = "Qwen2-1.5B-Instruct"
    elif ALG == 'mf_wo_sft_mf_a':
        a_MODEL_NAME = "Qwen2-1.5B-Instruct"
        mf_MODEL_NAME = "Qwen2-1.5B-Instruct"
    elif ALG == 'state_sft':
        a_MODEL_NAME = "qwen2-1.5B-state-0220"
        mf_MODEL_NAME = "Qwen2-1.5B-Instruct"
    else:
        print("ERROR ALG NAME!")
        a_MODEL_NAME = None
        mf_MODEL_NAME = None
    args.a_MODEL_NAME = a_MODEL_NAME
    args.mf_MODEL_NAME = mf_MODEL_NAME


    data = {
        "News": {
            # Existing
            "sewage": "3533780360431082.json",  # 帝都污水
            "weibo": "3472189711824480.json",  # 微博言论自由：rumor
            "traffic": "3455278454240838.json",  # 交通新规
            "fund": "3437038370353137.json",  # 基金亏损问题
            "bank": "13347733575.json",  # 银行职员收到短信
            "towing": "3910494551004498.json",  # 天价拖车
            # New
            "suicide": "3549913478403360.json",  # 宁波母子跳楼
            "child": "3552389523111081.json",  # 长春失踪孩子找到
            "collapse": "3554968256246645.json",  # 腾讯员工累倒传言
            "death": "3555026469159732.json",  # 腾讯员工暴毙反思
            "pollution": "3555670261721720.json",  # 悉尼vs上海水污染
            "protest": "3558766337240721.json"  # 洛阳裸女喊冤
        },
        "Sports": {
            # Existing
            "liuxiang": "3476720403686044.json",  # 刘翔
            "hengda": "3911688812086927.json",  # 亚冠决赛次回合 恒大首发
            "football": "3917810986061517.json"  # 足球
            # New: None (Sports remains empty in new dataset)
        },
        "Politics": {
            # Existing
            "retirement": "3454256721550776.json",  # 延迟退休政策
            "property": "3479678805595388.json",  # 房产税开征
            "wage": "3918809943738630.json",  # 最低工资
            # New
            "corruption": "3549735002472473.json",  # 广西局长吃509份低保
            "incense": "3557586148357412.json"  # 太原市长烧香
        },
        "Technology": {
            "phone": "3476379494883694.json",  # 发手机
            # New
            "apple": "3556418781490303.json",  # 央视315黑苹果
            "jac": "3556525094196431.json",  # 央视黑江淮车
            "vw": "3556597777008259.json",  # 央视收大众钱
            "visitors": "3558335318239831.json"  # 微博访客功能
        },
        "Health": {
            "water": "3466379833885944.json",  # 喝水谣言
            "durian": "3607987484215361.json",  # 吃榴莲谣言
            # New
            "explosion": "3549696167557624.json",  # 电暖宝爆炸
            "mengniu": "3551764454340038.json",  # 蒙牛2月30日生产
            "bug": "3557714938525397.json"  # 隐翅虫警示
        },
        "Crime": {
            # Existing
            "attack": "3909189702979876.json",  # 女子街头持刀砍人
            # New
            "rape": "3545636475076588.json",  # 轮奸12岁少女
            "murder": "3546008959383242.json",  # 李翔被杀
            "settlement": "3549679364631318.json",  # 李天一和解
            "hit": "3550671498093079.json",  # 王云岗撞人
            "smuggling": "3551712554354222.json",  # 人体藏奶粉
            "substitute": "3552907573807786.json",  # 周喜军顶罪
            "prostitution": "3552968810662725.json",  # 武汉高校卖淫
            "defense": "3553871978592378.json",  # 李天一不构成轮奸
            "crush": "3557585443383199.json",  # 深圳城管碾老人
            "vice": "3558061107430414.json"  # 副市长车碾警花 (renamed from "attack" to avoid clash)
        },
        "Culture": {
            # Existing
            "kangxi": "3464863219571406.json",  # 康熙十世孙
            "conspiracy": "3463394076393322.json",  # “隐秘聚会”阴谋论
            "lobster": "3466544342618479.json",  # 日本人不吃小龙虾
            "houge": "3555198087457574.json",  # 大圣西去 (西游记相关)
            # New
            "china": "3548612241976905.json",  # 感动中国
            "lyrics": "3549364918002421.json",  # 想你的夜歌词
            "meaning": "3552073255524560.json",  # 玩微博意义
            "festival": "3552919997329099.json",  # 摸奶节
            "joke": "3556530529722533.json",  # 何润东删稿
            "leslie": "3556608481082010.json"  # 张国荣死因
        }
    }

    file_class, test_files = find_file_info(file_name)
    args.file_class = file_class
    
    evaluation_results = runner(ALG,model_type, test_files, comment_n, a_MODEL_NAME, mf_MODEL_NAME, generate_true=True)

    # 保存结果
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if mf_MODEL_NAME == a_MODEL_NAME:
        path = mf_MODEL_NAME
    else:
        path = mf_MODEL_NAME + "/" + a_MODEL_NAME

    OUTPUT_PATH = f'save_data/{path}/evaluation/'
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    output_file = os.path.join(OUTPUT_PATH, f'{file_class}_{file_name}_{ALG}_start_{simulation_start}_n_{comment_n}_s_{seed_num}_evaluation_results_{current_time}.csv')
    evaluation_results.to_csv(output_file, index=False)
    print(f"评估结果已保存至 {output_file}")
    

    # 将 args 保存为字典并记录到 JSON 文件
    args_dict = vars(args)  # 将 argparse.Namespace 转换为字典
    args_dict['output_file'] = output_file  # 添加输出文件路径
    
    task = args.task
    if task != "":
        saved_paths_file = f'save_data/{task}/{model_type}/saved_file_paths.json'
    else:
        saved_paths_file = 'save_data/saved_file_paths.json'


    dir_path = os.path.dirname(saved_paths_file)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    # 如果文件已存在，读取现有内容
    if os.path.exists(saved_paths_file):
        with open(saved_paths_file, 'r') as f:
            saved_data = json.load(f)
    else:
        saved_data = []
    # 添加新的记录
    saved_data.append(args_dict)

    # 写回 JSON 文件
    with open(saved_paths_file, 'w') as f:
        json.dump(saved_data, f, indent=4)

    print(f"文件路径和参数已记录到 {saved_paths_file}")
