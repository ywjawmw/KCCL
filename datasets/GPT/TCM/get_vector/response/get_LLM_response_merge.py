# --*-- conding:utf-8 --*--
# @Time : 2024/12/11 15:46
# @Author : YWJ
# @Email : 52215901025@stu.ecnu.edu.cn
# @File : get_LLM_response.py
# @Software : PyCharm
# @Description : 根据KG的内容获得LLM的回复
import json
from Openai import OpenaiAPI
from bench_function import export_distribute_json, export_union_json
import os
from Qwen import QwenAPI

herb_pair_prompt = "你是一个资深的中药信息分析和总结专家，根据中药的功效、药味、归经、药性、化合物分子、部位、毒性、用量、用法、贮藏、治疗的症状以及与其他中药的关系等信息，总结中药<name>。要求总结的语言流畅，逻辑通顺。用一段话进行描述，不要分段，不能丢失输入的任何关系信息。"

sym_pair_prompt = "你是一个资深的症状信息分析专家，根据输入的与症状的相关的证候、治疗的中药、对应的英文症状以及与其他症状的关系等信息，总结症状<name>。要求总结的语言流畅，逻辑通顺。用一段话进行描述，不要分段，不能丢失输入的任何关系信息。"

# 读写json文件
def read_json(file_path, encoding='utf-8'):
    with open(file_path, 'r', encoding=encoding) as file:
        data = json.load(file)
    return data

def write_json(file_path, data, encoding='utf-8'):
    with open(file_path, 'w', encoding=encoding) as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

# 下面是头痛与其他症状之间的信息：

if __name__ == "__main__":
    # Load the FBQ_prompt.json file
    # os.environ['HTTPS_PROXY'] = "http://127.0.0.1:33210"

    dataset_name = "Herb"
    directory = f"../data/{dataset_name}"

    # get the model_name and instantiate model_api
    model_type = "Qwen"
    if model_type == "OpenAI":
        os.environ['HTTPS_PROXY'] = "http://127.0.0.1:10809"
        openai_api_key = "sk-XXX"  # key
        # os.environ['OPENAI_API_KEY'] = openai_api_key
        os.environ["OPENAI_BASE_URL"] = "https://api.xiaoai.plus/v1"
        model_name = 'gpt-3.5-turbo'
        model_api = OpenaiAPI(openai_api_key, model_name=model_name)
    elif model_type == "Qwen":
        os.environ['HTTPS_PROXY'] = "http://127.0.0.1:10809"
        # model_name = "qwen-max"
        # model_name = "qwen2.5-7b-instruct-1m"
        model_name = "deepseek-r1"
        model_api = QwenAPI(model_name=model_name)

    question_type = "entity_merge" # "entity_pair"
    print(model_name)
    print(question_type)

    keyword = "herb_merge"
    zero_shot_prompt_text = herb_pair_prompt
    print(keyword)

    export_distribute_json(
        model_api,
        model_name,
        directory,
        keyword,
        zero_shot_prompt_text,
        question_type,  # "entity_pair"、encoder
        entity_type="herb",
    )

    export_union_json(
        directory,
        model_name,
        keyword,
        zero_shot_prompt_text
    )


    ###########symptom

    keyword = "symptom_merge"
    zero_shot_prompt_text = sym_pair_prompt
    print(keyword)

    export_distribute_json(
        model_api,
        model_name,
        directory,
        keyword,
        zero_shot_prompt_text,
        question_type,  # "entity_pair"
        entity_type="symptom",
    )

    export_union_json(
        directory,
        model_name,
        keyword,
        zero_shot_prompt_text
    )

