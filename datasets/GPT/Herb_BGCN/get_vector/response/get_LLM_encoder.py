# --*-- conding:utf-8 --*--
# @Time : 2024/12/11 15:46
# @Author : YWJ
# @Email : 52215901025@stu.ecnu.edu.cn
# @File : get_LLM_response.py
# @Software : PyCharm
# @Description : 根据LLM的回复, 通过LLMs获得编码
import json
from Openai import OpenaiAPI
from bench_function import export_distribute_json_encoder, export_union_encoder
import os

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
    os.environ['HTTPS_PROXY'] = "http://127.0.0.1:10809"
    dataset_name = "Herb"
    for LLM_name in ["gpt-3.5-turbo"]:  # "gpt-4-0613", "gpt-4o-2024-05-13", "gpt-3.5-turbo"
        directory = f"../data/{dataset_name}"
        openai_api_key = "skXX" # text-embedding-3-large
        os.environ['OPENAI_API_KEY'] = openai_api_key
        os.environ["OPENAI_BASE_URL"] = "https://api.xiaoai.plus/v1"
        # get the model_name and instantiate model_api
        model_type = "OpenAI"
        if model_type == "OpenAI":
            model_name = 'text-embedding-3-large'
            model_api = OpenaiAPI(openai_api_key, model_name=model_name)

        print(model_name)
        #
        keyword = "herb_pair"
        print(keyword)

        export_distribute_json_encoder(
            model_api,
            model_name,
            directory,
            keyword,
            LLM_name,
        )

        export_union_encoder(
            directory,
            model_name,
            LLM_name,
            keyword

        )


        ############symptom

        # keyword = "symptom_merge_summary"
        # print(keyword)
        #
        # export_distribute_json_encoder(
        #     model_api,
        #     model_name,
        #     directory,
        #     keyword,
        #     LLM_name,
        # )
        #
        # export_union_encoder(
        #     directory,
        #     model_name,
        #     LLM_name,
        #     keyword
        #
        # )