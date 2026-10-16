# --*-- conding:utf-8 --*--
# @Time : 2024/12/13 16:46
# @Author : YWJ
# @Email : 52215901025@stu.ecnu.edu.cn
# @File : vector.py
# @Software : PyCharm
# @Description :  把没有信息的向量用0向量填充

import json
import pickle
import os

# 读写json文件
def read_json(file_path, encoding='utf-8'):
    with open(file_path, 'r', encoding=encoding) as file:
        data = json.load(file)
    return data

if __name__ == "__main__":
    entity_name = "herb"
    num = 753
    batch_size = 1
    example_num = num
    for LLM_name in ["gpt-4-0613", "gpt-3.5-turbo"]:  # "gpt-4o-2024-05-13",
        save_directory = f"text-embedding-3-large_{LLM_name}_{entity_name}_pair"
        files = [file for file in os.listdir(save_directory) if file.endswith('.json')]
        vector = []
        for idx in range(0, num):
            start_num = idx * batch_size
            end_num = min(start_num + batch_size, example_num)
            if start_num >= example_num:
                break
            file_name = f"seperate_{int(start_num)}-{int(end_num - 1)}.json"
            file_path = os.path.join(save_directory, file_name)
            # 判断文件是否存在
            if os.path.exists(file_path):
                # Load and merge the data from the JSON files
                with open(file_path, "r", encoding='utf-8') as f:
                    data = json.load(f)
                    vector.append(data['example'][0]['embedding'])
            else:
                vector.append([0.0] * 3072)
        print(len(vector))
        with open(f"vector_{LLM_name}_{entity_name}_pair.pkl", "wb") as file:
            pickle.dump(vector, file)
        print("Fill the empty vector successfully!")
