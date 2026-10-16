
import os
import json
import sys
import time
import re
# from random import choice
# import requests
from typing import List, Union, Dict
# from joblib import Parallel, delayed
import pickle
from tqdm import  tqdm



def get_api_key(filename: str, start_num: int, end_num: int) -> List[str]:
    """
    Retrieves API keys from a file.

    :param filename: Name of the file containing API keys
    :param start_num: Starting line number for reading the file
    :param end_num: Ending line number for reading the file
    :return: List of API keys
    """
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    pattern = re.compile(r'sk-[\s\S]*?(?=\s*\n)')
    api_key_list = []
    
    for i in range(start_num, end_num):
        api_key = pattern.findall(lines[i])
        if len(api_key) != 0:
            api_key_list.append(api_key[0])
    
    return api_key_list


def extract_choice_answer(model_output, question_type, answer_lenth=None):
    """
    Extract choice answer from model output

    Format of model_output that is expected:
    'single_choice': choice answer should be the last Capital Letter of the model_output, e.g.: "...【答案】 A <eoa>"
    'multi_question_choice': "...【答案】A ... 【答案】C ..." or write the choice answers at the beginning of the model_output, e.g. "A C D E F...."
    'multi_choice': "...【答案】 ABD " or write the choice answers at the end of the model_output, e.g. "... ACD"
    'five_out_of_seven': choice answers should be the first five Capital Letters of the model_output, e.g. "A C D F B ...."
    """
    if question_type == 'A1+A2' or question_type == 'A3+A4' or question_type == 'B1' or question_type == 'differ_option':
        model_answer = []
        # temp = re.findall(r'[A-E]', model_output[::-1])
        # if len(temp) != 0:
        #     model_answer.append(temp[0])
        model_output = model_output[::-1]
        pattern = r"([A-Z]).*?案答"
        check_info = re.search(pattern, model_output)
        if check_info:
            pattern = r"\．[A-Z]"
            temp = re.findall(pattern, model_output)
            if len(temp) > 0:
                # answer = temp[0]
                answer = check_info.group(1)
                model_answer.append(answer)
            else:
                temp = re.findall(r'[A-E]', model_output)
                if len(temp) != 0:
                    answer = temp[0]
                    model_answer.append(answer)
        else:
            temp = re.findall(r'[A-E]', model_output)
            if len(temp) != 0:
                answer = temp[0]
                model_answer.append(answer)
    elif question_type == "":
        model_answer = []
        model_output = model_output[0].replace("[", "").replace("]", "").replace("<eoa>", "").replace("</eoa>", "").replace(" ", "")
        model_output = ''.join([char for char in model_output if char.isdigit()])
        temp = re.findall(r'[0-9]', model_output)
        if len(temp) != 0:
            model_answer.append(model_output)

    elif question_type == 'multi_question_choice':
        model_answer = []
        temp = re.findall(r"【答案】\s*[:：]*\s*[A-Z]", model_output)
            
        if len(temp) == answer_lenth:
            for t in temp:
                model_answer.append(re.findall(r'[A-Z]', t)[0])
        else:
            temp = re.findall(r"[A-Z]", model_output)
            if len(temp) > 0:
                for k in range(min(len(temp), answer_lenth)):
                    model_answer.append(temp[k])
    elif question_type == "test_text":
        model_answer = []
        model_output = model_output[0].replace("[", "").replace("]", "").replace("<eoa>", "").replace("</eoa>", "").replace(" ", "")
        temp = model_output.split(",")
        if len(temp) != 0:
            if len(temp) < 20:
                # 在temp补若干个-1，让其长度为20
                for i in range(20-len(temp)):
                    temp.append("-1")
            if len(temp) > 20:
                temp = temp[:20]
            model_answer.append(temp)
    elif question_type == 'multi_choice':
        model_answer = []
        answer = ''
        content = re.sub(r'\s+', '', model_output)
        answer_index = content.find('【答案】')
        if answer_index > 0:
            temp = content[answer_index:]
            if len(re.findall(r'[A-E]', temp)) > 0:
                for t in re.findall(r'[A-E]', temp):
                    answer += t
        else:
            temp = content[-10:]
            if len(re.findall(r'[A-E]', temp)) > 0:
                for t in re.findall(r'[A-E]', temp):
                    answer += t
        if len(answer) != 0:
            model_answer.append(answer)
    
    elif question_type == 'five_out_of_seven':
        model_answer = []
        temp = re.findall(r'[A-G]', model_output)
        if len(temp) > 0:
            for k in range(min(5, len(temp))):
                model_answer.append(temp[k])

    return model_answer

def choice_test_A12(**kwargs):
    model_api = kwargs['model_api']
    model_name = kwargs['model_name']
    start_num = kwargs['start_num']
    end_num = kwargs['end_num']
    data = kwargs['data']['example']
    keyword = kwargs['keyword']
    prompt = kwargs['prompt']
    question_type = kwargs['question_type']
    save_directory = kwargs['save_directory']
   
    model_answer_dict = []
    for i in range(start_num, end_num):

        index = data[i]['index']
        question = data[i]['question'].strip() + '\n'
        # year = data[i]['year']
        # category = data[i]['year']
        score = data[i]['score']
        standard_answer = data[i]['answer']
        answer_lenth = len(standard_answer)
        analysis = data[i]['analysis']
        knowledge_point = data[i]['knowledge_point']
        model_output = model_api(prompt, question, "", question_type, None, 5)[0]   # list()
        model_answer = extract_choice_answer(model_output, question_type, answer_lenth)
        # # 重复回答5次
        # model_output_t0 = model_api[0](prompt, question, "", question_type, None, 1)
        # model_output_t_random = model_api[1](prompt, question, "", question_type, examples=None, answer_number=5)
        # model_output = model_output_t0 + model_output_t_random
        # model_answer = list()
        # for model_o in model_output:
        #     model_a = extract_choice_answer(model_o, question_type, answer_lenth)
        #     model_answer.append(model_a)
        # TODO: which content of temp we expect
        dict = {
            'index': index, 
            # 'year': year,
            # 'category': category,
            'score': score,
            'question': question, 
            'standard_answer': standard_answer,
            'analysis': analysis,
            'knowledge_point': knowledge_point,
            'model_answer': model_answer,
            'model_output': model_output
        }
        print("*" * 100, "index-", dict["index"], "*" * 100)
        for key, value in dict.items():
            print(key, ":", value)
        # print(dict)
        model_answer_dict.append(dict)

    file_name = model_name+"_seperate_"+keyword+f"_{start_num}-{end_num-1}.json"
    file_path = os.path.join(save_directory, file_name)
    with open(file_path, 'w', encoding='utf-8') as f:
        output = {
            'keyword': keyword, 
            'example': model_answer_dict
            }
        json.dump(output, f, ensure_ascii=False, indent=4)
        f.close()
def choice_test_TCM_Rec(**kwargs):
    model_api = kwargs['model_api']
    model_name = kwargs['model_name']
    start_num = kwargs['start_num']
    end_num = kwargs['end_num']
    data = kwargs['data']['example']
    keyword = kwargs['keyword']
    prompt = kwargs['prompt']
    save_directory = kwargs['save_directory']
    question_type = kwargs['question_type']

    model_answer_dict = []
    for i in range(start_num, end_num):
        question = "、".join(data[i]['sym_set'])
        # option = data[i]['option']
        standard_answer = data[i]['herb_set']
        model_output = model_api(prompt, question, "", question_type, "", 0)
        model_answer = extract_choice_answer(model_output, "test_text", 0)[0]
        # TODO: which content of temp we expect
        dict = {
            'sym_set': data[i]['sym_set'],
            'model_output': model_answer,
            'herb_set': standard_answer,
        }
        # print("*" * 100, "index-", dict["index"], "*" * 100)
        for key, value in dict.items():
            print(key, ":", value)
        # print(dict)
        model_answer_dict.append(dict)

    file_name = model_name + "_seperate_" + keyword + f"_{start_num}-{end_num - 1}.json"
    file_path = os.path.join(save_directory, file_name)
    with open(file_path, 'w', encoding='utf-8') as f:
        output = {
            'keyword': keyword,
            'example': model_answer_dict
        }
        json.dump(output, f, ensure_ascii=False, indent=4)
        f.close()

def choice_entity_pair(**kwargs):
    model_api = kwargs['model_api']
    model_name = kwargs['model_name']
    start_num = kwargs['start_num']
    end_num = kwargs['end_num']
    data = kwargs['data']['example']
    keyword = kwargs['keyword']
    prompt = kwargs['prompt']
    entity_type = kwargs['entity_type']
    question_type = kwargs['question_type']
    save_directory = kwargs['save_directory']

    model_answer_dict = []
    for i in range(start_num, end_num):
        entity_id = data[i][f'{entity_type}_id']
        entity_name = data[i][f'{entity_type}_name']
        entity_description = data[i][f'{entity_type}_description']
        if len(entity_description[0]) > 0:
            sys_entity_prompt = entity_description[-1]
            user_entity_prompt = "\n".join(entity_description[:-1])
            if entity_type == "herb":
                e_t = "中药"
            elif entity_type == "symptom":
                e_t = "症状"
            else:
                e_t = ""
                sys.exit()
            user_entity_prompt = f"下面是{entity_name}与其他{e_t}之间的信息：\n{user_entity_prompt}"
            sys_prompt = prompt.replace("<input_info>", sys_entity_prompt)
            model_output = model_api(sys_prompt, user_entity_prompt, question_type)
        else:
            model_output = ""
        # TODO: which content of temp we expect
        dict = {
            'id': int(entity_id),
            'name': entity_name,
            'pair_summary': model_output
        }
        # print("*" * 100, "index-", dict["index"], "*" * 100)
        for key, value in dict.items():
            print(key, ":", value)
        # print(dict)
        model_answer_dict.append(dict)

    file_name = f"seperate_{start_num}-{end_num - 1}.json"
    file_path = os.path.join(save_directory, file_name)
    with open(file_path, 'w', encoding='utf-8') as f:
        output = {
            'keyword': keyword,
            'example': model_answer_dict
        }
        json.dump(output, f, ensure_ascii=False, indent=4)
        f.close()


def choice_entity_merge(**kwargs):
    model_api = kwargs['model_api']
    model_name = kwargs['model_name']
    start_num = kwargs['start_num']
    end_num = kwargs['end_num']
    data = kwargs['data']['example']
    keyword = kwargs['keyword']
    prompt = kwargs['prompt']
    entity_type = kwargs['entity_type']
    question_type = kwargs['question_type']
    save_directory = kwargs['save_directory']

    model_answer_dict = []
    for i in range(start_num, end_num):
        entity_id = data[i]['id']
        entity_name = data[i]['name']
        entity_description = data[i]['description']
        if len(entity_description) > 0:
            if entity_type == "herb":
                e_t = "相关的功效、药味、归经、药性、化合物分子、部位、毒性、用量、用法、贮藏、治疗的症状以及与其他中药的关系等信息"
            elif entity_type == "symptom":
                e_t = "相关的证候、治疗的中药、对应的英文症状以及与其他症状的关系等信息"
            else:
                e_t = ""
                sys.exit()
            user_entity_prompt = f"下面是和{entity_name}{e_t}：\n{entity_description}"
            sys_prompt = prompt.replace("<name>", entity_name)
            model_output = model_api(sys_prompt, user_entity_prompt, question_type)
        else:
            model_output = ""
        # TODO: which content of temp we expect
        dict = {
            'id': entity_id,
            'name': entity_name,
            'summary': model_output
        }
        # print("*" * 100, "index-", dict["index"], "*" * 100)
        for key, value in dict.items():
            print(key, ":", value)
        # print(dict)
        model_answer_dict.append(dict)

    file_name = f"seperate_{start_num}-{end_num - 1}.json"
    file_path = os.path.join(save_directory, file_name)
    with open(file_path, 'w', encoding='utf-8') as f:
        output = {
            'keyword': keyword,
            'example': model_answer_dict
        }
        json.dump(output, f, ensure_ascii=False, indent=4)
        f.close()

def choice_Encoder(**kwargs):
    model_api = kwargs['model_api']
    start_num = kwargs['start_num']
    end_num = kwargs['end_num']
    data = kwargs['data']['example']
    keyword = kwargs['keyword']
    save_directory = kwargs['save_directory']

    model_answer_dict = []
    for i in range(start_num, end_num):
        entity_id = data[i]['id']
        entity_name = data[i]['name']
        # entity_description = data[i]['summary']
        entity_description = data[i]['pair_summary']
        if len(entity_description) > 0:
            model_output = model_api("", entity_description, "encoder")
        else:
            model_output = [0.0] * 3072
        # TODO: which content of temp we expect
        dict = {
            'id': entity_id,
            'name': entity_name,
            'embedding': model_output
        }
        # print("*" * 100, "index-", dict["index"], "*" * 100)
        for key, value in dict.items():
            print(key, ":", value)
        # print(dict)
        model_answer_dict.append(dict)
        file_name = f"seperate_{start_num}-{end_num - 1}.json"
        file_path = os.path.join(save_directory, file_name)
        with open(file_path, 'w', encoding='utf-8') as f:
            output = {
                'keyword': keyword,
                'example': model_answer_dict
            }
            json.dump(output, f, ensure_ascii=False, indent=4)
            f.close()

def export_union_encoder(directory: str, model_name:str, LLM_name: str, keyword: str) -> None:
    """
    Merges JSON files containing processed examples in a directory into a single JSON file.

    :param directory: Directory containing the JSON files
    :param model_name: Name of the model used to process the examples
    :param keyword: Keyword used to identify the JSON files
    """
    vector = []
    save_directory = os.path.join(directory, f'{model_name}_{LLM_name}_{keyword}')
    if os.path.exists(save_directory):
        output = {
                        'keyword': f'{LLM_name}_{keyword}',
                        'model_name': model_name,
                        'example': []
                    }

        # Iterate through the JSON files with the specified keyword in the directory

        print("Start to merge json files")
        files = [file for file in os.listdir(save_directory) if file.endswith('.json')]
        # print("Start to merge json files")
        example_num = len(files)
        parallel_num = example_num
        batch_size = example_num / parallel_num
        for idx in range(0, parallel_num):
            # print("*" * 100, idx)
            start_num = idx * batch_size
            end_num = min(start_num + batch_size, example_num)
            if start_num >= example_num:
                break
            file_name = f"seperate_{int(start_num)}-{int(end_num-1)}.json"
            file_path = os.path.join(save_directory, file_name)
            # 判断文件是否存在
            if os.path.exists(file_path):
                # Load and merge the data from the JSON files
                with open(file_path, "r", encoding='utf-8') as f:
                    data = json.load(f)
                    if len(data['example'][0]['embedding']) > 0:
                        output['example'] += (data['example'])
                        vector.append(data['example'][0]['embedding'])

        # Save the merged data into a single JSON file
        # merge_file = os.path.join(directory, f'Label_Qwen_Analysis/{model_name}_{keyword}.json')
        merge_file = os.path.join(directory, f'{model_name}_{LLM_name}_{keyword}.json')
        # output['example'] = sorted(output['example'], key=lambda x: x['index'])
        with open(merge_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
            print(merge_file)
        print(merge_file)
        # Save the merged data into a single pkl file
        vector_file = os.path.join(directory, f'vector_{LLM_name}_{keyword}.pkl')
        with open(vector_file, 'wb') as f:
            pickle.dump(vector, f)
            print(vector_file)


def export_union_json(directory: str, model_name: str, keyword: str, zero_shot_prompt_text: str) -> None:
    """
    Merges JSON files containing processed examples in a directory into a single JSON file.

    :param directory: Directory containing the JSON files
    :param model_name: Name of the model used to process the examples
    :param keyword: Keyword used to identify the JSON files
    :param zero_shot_prompt_text: Prompt text for zero-shot learning
    :param question_type: Type of questions in the JSON files (e.g. single_choice, five_out_of_seven, etc.)
    """

    save_directory = os.path.join(directory, f'{model_name}_{keyword}_summary')
    # save_directory = os.path.join(directory, f'{model_name}_{keyword}')  # herb_pair
    if os.path.exists(save_directory):
        output = {
                        'keyword': keyword,
                        'model_name': model_name,
                        'prompt': zero_shot_prompt_text,
                        'example': []
                    }

        # Iterate through the JSON files with the specified keyword in the directory

        print("Start to merge json files")
        files = [file for file in os.listdir(save_directory) if file.endswith('.json')]
        # print("Start to merge json files")
        example_num = len(files)
        parallel_num = example_num
        batch_size = example_num / parallel_num
        for idx in range(0, parallel_num):
            # print("*" * 100, idx)
            start_num = idx * batch_size
            end_num = min(start_num + batch_size, example_num)
            if start_num >= example_num:
                break
            file_name = f"seperate_{int(start_num)}-{int(end_num-1)}.json"
            file_path = os.path.join(save_directory, file_name)
            # Load and merge the data from the JSON files
            with open(file_path, "r", encoding='utf-8') as f:
                data = json.load(f)
                output['example'] += (data['example'])

        # Save the merged data into a single JSON file
        # merge_file = os.path.join(directory, f'{model_name}_{keyword}.json') # herb_pair
        merge_file = os.path.join(directory, f'{model_name}_{keyword}_summary.json')
        # output['example'] = sorted(output['example'], key=lambda x: x['index'])
        with open(merge_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
            print(merge_file)
        print(merge_file)

def export_distribute_json(
        model_api,
        model_name: str, 
        directory: str, 
        keyword: str, 
        zero_shot_prompt_text: str or List[str], 
        question_type: str,
        entity_type: str
    ) -> None:
    """
    Distributes the task of processing examples in a JSON file across multiple processes.

    :param model_name: Name of the model to use
    :param directory: Directory containing the JSON file
    :param keyword: Keyword used to identify the JSON file
    :param zero_shot_prompt_text: Prompt text for zero-shot learning
    :param question_type: Type of questions in the JSON file (e.g. single_choice, five_out_of_seven, etc.)
    :param entity_type: Type of entity in the JSON file (e.g. herb, symptom, etc.)
    """
    # Find the JSON file with the specified keyword
    for root, _, files in os.walk(directory):
        for file in files:
            if file == f'{model_name}_{keyword}.json':
            # if file == f'{keyword}.json':   # herb_pair
                filepath = os.path.join(root, file)
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
    
    example_num = len(data['example'])
    parallel_num = example_num
        
    # Prepare the list of keyword arguments for parallel processing
    kwargs_list = []
    batch_size = example_num // parallel_num
    save_directory = os.path.join(directory, f'{model_name}_{keyword}_summary')  # herb_merge
    # save_directory = os.path.join(directory, f'{model_name}_{keyword}') # herb_pair
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    # os.system(f'mkdir {save_directory}')

    for idx in range(81, parallel_num):
        start_num = idx * batch_size
        end_num = min(start_num + batch_size, example_num)
        if start_num >= example_num:
            break

        kwargs = {
            'model_api': model_api,
            'start_num': start_num,
            'end_num': end_num,
            'model_name': model_name, 
            'data': data, 
            'keyword': keyword, 
            'prompt': zero_shot_prompt_text, 
            'question_type': question_type,
            'entity_type': entity_type,
            'save_directory': save_directory,
        }
        kwargs_list.append(kwargs)
    
    # Run parallel processing based on the question type
    if question_type in ["entity_pair"]:
        for kwargs in kwargs_list:
           choice_entity_pair(**kwargs)
    elif question_type == "entity_merge":
        for kwargs in kwargs_list:
            choice_entity_merge(**kwargs)
    elif question_type == "test_text":
        for kwargs in kwargs_list:
            choice_test_TCM_Rec(**kwargs)


def export_distribute_json_encoder(
        model_api,
        model_name: str,
        directory: str,
        keyword: str,
        LLM_name: str or List[str]
) -> None:
    """
    Distributes the task of processing examples in a JSON file across multiple processes.

    :param model_name: Name of the model to use
    :param directory: Directory containing the JSON file
    :param keyword: Keyword used to identify the JSON file
    :param zero_shot_prompt_text: Prompt text for zero-shot learning
    :param question_type: Type of questions in the JSON file (e.g. single_choice, five_out_of_seven, etc.)
    :param entity_type: Type of entity in the JSON file (e.g. herb, symptom, etc.)
    """
    # Find the JSON file with the specified keyword
    for root, _, files in os.walk(directory):
        for file in files:
            if file == f'{LLM_name}_{keyword}.json':
                filepath = os.path.join(root, file)
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

    example_num = len(data['example'])
    parallel_num = example_num

    # Prepare the list of keyword arguments for parallel processing
    kwargs_list = []
    batch_size = example_num // parallel_num
    save_directory = os.path.join(directory, f'{model_name}_{LLM_name}_{keyword}')
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    # os.system(f'mkdir {save_directory}')

    for idx in range(0, parallel_num):
        start_num = idx * batch_size
        end_num = min(start_num + batch_size, example_num)
        if start_num >= example_num:
            break

        kwargs = {
            'model_api': model_api,
            'start_num': start_num,
            'end_num': end_num,
            'model_name': model_name,
            'data': data,
            'keyword': keyword,
            'LLM_name': LLM_name,
            'save_directory': save_directory,
        }
        kwargs_list.append(kwargs)

    # Run parallel processing based on the question type
    for kwargs in kwargs_list:
        choice_Encoder(**kwargs)