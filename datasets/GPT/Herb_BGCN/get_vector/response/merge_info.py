# --*-- conding:utf-8 --*--
# @Time : 2024/12/13 14:51
# @Author : YWJ
# @Email : 52215901025@stu.ecnu.edu.cn
# @File : merge_info.py
# @Software : PyCharm
# @Description : 把LLM生成的pair的描述和entity的其他描述合并为一个文件

import json
import os

# 读写json文件
def read_json(file_path, encoding='utf-8'):
    with open(file_path, 'r', encoding=encoding) as file:
        data = json.load(file)
    return data

def write_json(file_path, data, encoding='utf-8'):
    with open(file_path, 'w', encoding=encoding) as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


# 读入两个json文件，合并为一个json文件

def merge_json(file_path1, file_path2, output_file, entity_name):
    data1 = read_json(file_path1)
    data2 = read_json(file_path2)
    data3 = {
        "keyword": "herb",
        "example": []
    }
    for herb_pair_des, herb_des in zip(data1["example"], data2["example"]):
        if len(herb_pair_des['pair_summary']) > 0:
            herb_entity_des = f"{herb_pair_des['pair_summary']}\n {herb_des[f'{entity_name}_description']}"
        else:
            herb_entity_des = f"{herb_des[f'{entity_name}_description']}"
        data = {
            "id": herb_pair_des["id"],
            "name": herb_pair_des["name"],
            "description": herb_entity_des
        }
        data3["example"].append(data)
    write_json(output_file, data3)


if __name__ == "__main__":
    # Load the FBQ_prompt.json file
    entity_name = "herb"
    file_path2 = f"../data/Herb/{entity_name}_description.json"
    for LLM_name in ["deepseek-r1"]: # "gpt-4-0613", "gpt-4o-2024-05-13", "gpt-3.5-turbo", "qwen-max", "qwen2.5-7b-instruct-1m"
        directory = f"../data/Herb"
        file_path1 = f"{directory}/{LLM_name}_{entity_name}_pair.json"
        output_file = f"../data/Herb/{LLM_name}_{entity_name}_merge.json"
        merge_json(file_path1, file_path2, output_file, entity_name)
        print("Merge successfully!")