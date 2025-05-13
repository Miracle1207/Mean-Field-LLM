import os
import json
import random
from datasets import Dataset

class DatasetLoader:
    def __init__(self, train_dir, seed):
        """Initialize the DatasetLoader with directories and a seed."""
        self.train_dir = train_dir
        self.test_dir = train_dir.replace("train", "test")
        self.seed = seed

    def build_state(self, item):
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

    def load_and_process_files(self, directory, num_files):
        """Load and process a specified number of files from a directory."""
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        random.seed(self.seed)
        selected_files = random.sample(files, min(num_files, len(files)))

        data = []
        for file_name in selected_files:
            file_path = os.path.join(directory, file_name)
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_data = json.load(f)

                # Sort data by time
                sorted_data = sorted(file_data, key=lambda x: x.get("t", 0), reverse=False)
                topic = sorted_data[0].get("text", "未知主题") if sorted_data else "未知主题"
                
                for idx, item in enumerate(sorted_data):
                    state = self.build_state(item)
                    # prompt = (
                    #     f"当前讨论的主题是：{topic}"
                    #     f"网友描述是 {state}"
                    # )
                    prompt = {
                        "topic": topic,
                        "state": state
                    }
                    response = item.get("text", "")
                    data.append({
                        "file_name": file_name,
                        "idx": idx,
                        "question": prompt, "response": response})

        return Dataset.from_list(data)

    def load_dataset(self, train_n, test_n):
        """Load training and testing datasets."""
        train_data = self.load_and_process_files(self.train_dir, train_n)
        test_data = self.load_and_process_files(self.test_dir, test_n)
        return train_data, test_data

# if __name__ == "__main__":
#     # Example usage
#     train_dir = "/home/mqr/code/mean_field_lm/data/rumdect/Weibo/train"  # Replace with the actual path to your training directory
#     seed = 42
#     loader = DatasetLoader(train_dir, seed)
#
#     train_n = 5  # Number of training files to load
#     test_n = 3   # Number of testing files to load
#
#     dataset = loader.load_dataset(train_n, test_n)
#
#     print("Training Data:")
#     for item in dataset["train"]:
#         print(item)
#
#     print("\nTesting Data:")
#     for item in dataset["test"]:
#         print(item)
