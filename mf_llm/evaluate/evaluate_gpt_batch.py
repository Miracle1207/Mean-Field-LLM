import os
import pandas as pd
# import openai  # 需要先 pip install openai
from openai import OpenAI
import json
import re
# ============== 请在此处替换为你的 OpenAI API Key ==============

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), )

# ============================================================
def build_prompt(topic, comments):
    """
    构建 GPT 所需的 Prompt 文本，将被讨论的主题和批量评论组合成完整的指令。

    Args:
        topic (str): 被讨论的主题
        comments (List[str]): 当前批次的评论列表

    Returns:
        str: 生成的完整 Prompt 字符串
    """
    prompt_header = f"""请你扮演资深舆论内容分析专家。现在有多条网友的评论，请对评论进行客观地分析，以评估网友评论中对下述话题的情感、立场及其观点意向。

    讨论话题是：{topic}\n
    任务要求：
    请对每条评论从以下 9 个维度进行分析，并严格按照 JSON 格式返回结果。特例："@_@" 属于颜文字，常表示“惊讶、困惑”或“呆住了”.

    1. rumor（谣言传播）从 `["spread", "counter"]` 选择。
       - `"counter"`: 该评论意在辟谣，不相信当前讨论的话题，包括具有依据的阐述事实、怀疑讨论内容的真实性、期待真相和直接指出其虚假性等，具有辟谣的作用。
       - `"spread"`：不属于 `"counter"` 的所有评论。包括因为相信话题是真的而表达的言论、转发微博、@用户名、重复话题内容、对话题内容表达情绪都属于传播行为。

    2. sentiment_state（情绪状态: 从评论中（包括标点符号），捕捉网友的情绪状态。特别注意是否有隐含的讽刺、批评性含义等。）
       - 从 `["angry", "calm", "happy", "sad", "fear", "surprise"]` 选择。转发微博属于 calm。

    3. sentiment_tendency（情绪倾向:从网友评论中，分析网友情绪是倾向于积极、消极或中立?）
       - 从 `["positive", "negative", "neutral"]` 选择。请特别注意是否有隐含的消极情绪。

    4. behavior_type（行为类型：如果内容是转发，则属于 share；如果内容是评价，则属于 comment.）
       - 从 `["comment", "share"]` 选择。

    5. stance（立场：根据评论的内容，分析网友对被讨论话题的立场，是支持还是反对？特别注意评论中是否隐含不满、反对、批判的含义。）
       - 从 `["support", "oppose", "neutral"]` 选择。

    6. belief_degree（评论对被讨论话题的真实性是相信(包括转发微博)，还是怀疑（包括期待真相、明确认为信息虚假等）？）
       - 从 `["believe", "doubt"]` 选择。

    7. keywords（关键词提取）
       - 以数组形式返回关键词，例如 `["政策", "经济"]`。如果评论无意义，则返回 `[""]`。

    8. subjectivity（该评论是基于主观观点还是客观事实？）
       - 从 `["subjective", "objective"]` 选择。

    9. intent_classification（评论的意图）从 `["question", "promotion", "opinion"]` 选择。
       - `"question"`：该评论意在提出问题；
       - `"promotion"`：该评论意在传播信息；
       - `"opinion"`：该评论意在发表看法和观点。

    """

    # 将评论内容按序号拼接
    comments_part = f"{len(comments)}个网友评论如下：\n"
    for idx, c in enumerate(comments, start=1):
        comments_part += f"第{idx}个网友的评论是：\"{c}\"\n"
    
    # 给出示例示意，示例一定要是规范 JSON
    # 注意：示例仅作格式参考，GPT 输出时只能给出最终数组，而非“json{...}”之类的内容
    prompt_middle = f"""
    输出格式：
    请严格按照以下 JSON 数组格式返回对{len(comments)}个评论的评估，不要输出除 JSON 以外的任何内容："""
    prompt_footer ="""
    ```json
    [
      {
        "rumor": "spread",
        "sentiment_state": "calm",
        "sentiment_tendency": "neutral",
        "behavior_type": "share",
        "stance": "neutral",
        "belief_degree": "believe",
        "keywords": ["转发", "微博"],
        "subjectivity": "objective",
        "intent_classification": "promotion",
      }
      {
        "rumor": "counter",
        "sentiment_state": "angry",
        "sentiment_tendency": "negative",
        "behavior_type": "comment",
        "stance": "oppose",
        "belief_degree": "doubt",
        "keywords": ["假", "不可能"],
        "subjectivity": "subjective",
        "intent_classification": "opinion",
      }
    ] """
    
    # 组合完整的 Prompt
    full_prompt = prompt_header + comments_part +prompt_middle+ prompt_footer
    return full_prompt


import json
import re


def extract_json(gpt_output):
    """
    尝试从 GPT 的输出文本中解析出 JSON 格式数据。
    为了尽可能提高成功率，本函数会依次采用多种方式进行提取和解析：
        1. 整体尝试解析
        2. 提取 ```json ...``` 代码块
        3. 提取任意 ```...``` 代码块
        4. 从第一处 '{' 到最后一处 '}' 截取并解析

    Args:
        gpt_output (str): GPT 返回的原始字符串

    Returns:
        Any: 解析得到的 JSON 数据（dict 或 list 等），若均失败则返回 None
    """
    
    # 1. 尝试整体解析
    try:
        return json.loads(gpt_output)
    except Exception:
        pass
    
    # 2. 尝试提取 ```json ...``` 代码块
    pattern_json_block = r"```json\s*([\s\S]*?)\s*```"
    match_json_block = re.search(pattern_json_block, gpt_output, re.IGNORECASE)
    if match_json_block:
        json_str = match_json_block.group(1).strip()
        try:
            return json.loads(json_str)
        except Exception:
            pass
    
    # 3. 尝试提取任意的 ``` ... ``` 代码块
    pattern_code_block = r"```([\s\S]*?)```"
    match_code_block = re.search(pattern_code_block, gpt_output)
    if match_code_block:
        json_str = match_code_block.group(1).strip()
        try:
            return json.loads(json_str)
        except Exception:
            pass
    
    # 4. 从文本中找到首个 '{' 和最后一个 '}'，提取中间内容尝试解析
    start_index = gpt_output.find("{")
    end_index = gpt_output.rfind("}")
    if start_index != -1 and end_index != -1 and start_index < end_index:
        possible_json_str = gpt_output[start_index:end_index + 1].strip()
        try:
            return json.loads(possible_json_str)
        except Exception:
            pass
    
    # 如果以上所有方式都失败，则返回 None
    return None


def call_gpt_for_evaluation(topic, comments,cache_path=None, batch_size=5):
    """
    给定一个话题和多条评论，通过 GPT 分批进行多维度分析并返回解析后的结果。

    Args:
        topic (str): 被讨论的主题
        comments (List[str]): 评论列表
        batch_size (int): 每个批次处理的评论数

    Returns:
        List[dict]: 解析后的 GPT 返回结果，每条评论一个 dict，包含多维度信息。
    """
    all_results = []

    # 将评论按照 batch_size 分段
    for start_idx in range(0, len(comments), batch_size):
        batch_comments = comments[start_idx : start_idx + batch_size]

        # 使用我们封装好的 Prompt 构建函数
        prompt = build_prompt(topic, batch_comments)

        try:
            gpt_data = []  # 初始化 gpt_data
            attempt = 0  # 记录尝试次数
    
            while len(gpt_data) != len(batch_comments):
                response = client.chat.completions.create(
                    model="gpt-4o-mini",  # 或者 "gpt-3.5-turbo"
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0,
                )
        
                gpt_output = response.choices[0].message.content
                gpt_data = extract_json(gpt_output)
                attempt += 1
        
                if attempt == 2 and len(gpt_data) != len(batch_comments):
                    print("Warning: GPT output length mismatch after 2 attempts.")
                    missing_count = len(batch_comments) - len(gpt_data)
                    gpt_data.extend([{}] * missing_count)  # 使用空 dict 补充
                    break  # 跳出循环，避免死循环
    
            all_results.extend(gpt_data)
            print(gpt_data)


        except json.JSONDecodeError:
            print("JSON 解析失败。GPT 输出可能包含非 JSON 格式内容。原始输出：\n", gpt_output)
        except Exception as e:
            print("调用 GPT 出错:", e)
    if cache_path != None:
        df = pd.DataFrame(all_results)
        df.to_csv( cache_path.replace("_eval.csv", "_gpt_cache.csv"), index=False, encoding="utf-8")
        print(f"结果已保存至 {cache_path.replace('_eval.csv', '_gpt_cache.csv')}")

    return all_results



import pandas as pd
import os
def evaluate_real_comments(df,unique_real_comments):
    if "topic" in df.columns:
        example_topic = str(df["topic"].iloc[0])
    else:
        example_topic = ""
    
    real_gpt_results = call_gpt_for_evaluation(example_topic, unique_real_comments)
    
    # 将 GPT 结果保存到一个 DataFrame 并写入 real_eval_csv
    real_eval_list = []
    for comment_text, analysis in zip(unique_real_comments, real_gpt_results):
        real_eval_list.append({
            "real_comment": comment_text,
            "real_rumor": analysis.get("rumor", None),
            "real_sentiment_state": analysis.get("sentiment_state", None),
            "real_sentiment_tendency": analysis.get("sentiment_tendency", None),
            "real_behavior_type": analysis.get("behavior_type", None),
            "real_stance": analysis.get("stance", None),
            "real_belief_degree": analysis.get("belief_degree", None),
            "real_keywords": analysis.get("keywords", None),
            "real_subjectivity": analysis.get("subjectivity", None),
            "real_intent_classification": analysis.get("intent_classification", None),
            # "real_topic_relevance": analysis.get("topic_relevance", None)
        })
    
    real_eval_df = pd.DataFrame(real_eval_list)
    return real_eval_df


import os
import pandas as pd


def evaluate_similarity_by_file(
        input_csv,
        real_eval_csv="real_comments_eval_new.csv",
        output_csv=None,
        simulation_start=50,
        evaluate_len=200
):
    """
    评估生成的评论，并与真实评论进行对比。
    """
    import os
    print("\n\n=================================================================")
    print(f"Processing file: {input_csv}")
    df = pd.read_csv(input_csv).fillna('转发微博').dropna(subset=['file'])[:evaluate_len]
    evaluate_len = min(evaluate_len, len(df))
    output_csv = output_csv or input_csv.replace(".csv", "_eval.csv")

    real_eval_dir = os.path.dirname(real_eval_csv)

    # 检查目录是否存在，如果不存在则创建
    if not os.path.exists(real_eval_dir):
        os.makedirs(real_eval_dir)
        print(f"目录 {real_eval_dir} 不存在，已创建。")

    if not os.path.exists(real_eval_csv):
        real_eval_df = evaluate_real_comments(df, df["real_comment"].tolist())
        real_eval_df.to_csv(real_eval_csv, index=False, encoding='utf-8-sig')
        print(f"{real_eval_csv} 不存在，已评估并保存。")
    else:
        real_eval_df = pd.read_csv(real_eval_csv)
        if len(real_eval_df) < evaluate_len:
            print(f"{real_eval_csv} 不完整，补充评估...")
            new_real_eval_df = evaluate_real_comments(df, df["real_comment"][len(real_eval_df): evaluate_len].tolist())
            real_eval_df = pd.concat([real_eval_df, new_real_eval_df], ignore_index=True)
            real_eval_df.to_csv(real_eval_csv, index=False, encoding='utf-8-sig')
            print(f"已更新 {real_eval_csv}，新增 {len(new_real_eval_df)} 条数据。")
    
    cached_real_analysis = {idx: row for idx, row in real_eval_df.iterrows()}
    first_mismatch_idx = next(
        (idx for idx in range(len(df)) if df["generated_comment"].iloc[idx] != df["real_comment"].iloc[idx]),
        simulation_start)
    
    exist_gen_df = pd.read_csv(output_csv) if os.path.exists(output_csv) else None
    cached_exist_gen_analysis = {idx: row for idx, row in exist_gen_df.iterrows()} if os.path.exists(output_csv) else None
    exist_gen_len = len(exist_gen_df) if exist_gen_df is not None else first_mismatch_idx
    
    if exist_gen_len >= evaluate_len:
        print("已达到评测长度，请检查文件！")
        return None
    
    print(f"还有 {evaluate_len - exist_gen_len} 条评论需要评测，开始评测...")
    topic = df["topic"].iloc[0] if "topic" in df.columns else ""
    gen_gpt_results = call_gpt_for_evaluation(topic, df["generated_comment"].iloc[exist_gen_len:evaluate_len].tolist(),cache_path=output_csv)
    
    all_results = []
    for idx, row in df.iterrows():
        
        real_analysis = cached_real_analysis.get(idx, {})
        if idx < exist_gen_len and idx < first_mismatch_idx:
            gen_analysis = real_analysis.rename(lambda x: x.replace("real_", "") if x.startswith("real_") else x)
            label = ""
        elif idx < exist_gen_len and idx >= first_mismatch_idx:
            gen_analysis = cached_exist_gen_analysis.get(idx, {})
            label = "gen_"
        else:
            gen_analysis = gen_gpt_results[idx - exist_gen_len] if idx - exist_gen_len < len(gen_gpt_results) else {}
            label = ""
        
        row_dict = row.to_dict()
        for prefix in ["real", "gen"]:
            for key in ["rumor", "sentiment_state", "sentiment_tendency", "behavior_type", "stance", "belief_degree",
                        "keywords", "subjectivity", "intent_classification"]:
                row_dict[f"{prefix}_{key}"] = real_analysis.get(f"real_{key}", None) if prefix == "real" else gen_analysis.get(f"{label}{key}",None)
        all_results.append(row_dict)
    
    result_df = pd.DataFrame(all_results)

    result_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"所有 GPT 分析结果已保存到 {output_csv}")
    return result_df


import json
from multiprocessing import Pool
from functools import partial


def get_file_info(eval_dir_path):
    results = []
    with open(eval_dir_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for item in data:
            file_info = {
                'file_name': item['file_name'],
                'output_file': item['output_file']
            }
            results.append(file_info)
    return results


def eval_gpt(file_info):
    file_name = file_info['file_name']
    file_path = f"../scripts/{file_info['output_file']}"
    real_eval_path = f"../scripts/generated_data/real_data/real_comments_eval_{file_name}.csv"
    
    print(f"Processing: file_name: {file_name}")
    print(f"output_file: {file_path}")
    
    evaluate_similarity_by_file(
        input_csv=file_path,
        real_eval_csv=real_eval_path,
        evaluate_len=300
    )


if __name__ == "__main__":
    eval_dir_path = "../scripts/save_data/main/saved_file_paths.json"
    file_data = get_file_info(eval_dir_path)
    
    # 使用多进程池并行执行
    with Pool(processes=4) as pool:  # 可以根据你的 CPU 核心数调整 processes 的值
        pool.map(eval_gpt, file_data)

    
    