import random
import sys

import requests
import time
import openai
import os
from random import choice


class OpenaiAPI:
    def __init__(self, api_key_list:str, model_name:str="gpt-3.5-turbo", temperature:float=0.0, max_tokens: int=1024):
        self.api_key_list = api_key_list
        self.model_name = model_name    # 新的model, 支持1w+
        self.temperature = temperature
        self.max_tokens = max_tokens   # 不设置

    def send_request_encoder(self, request_text: str) -> str:
        """
        """
        while True:
            try:
                os.environ['HTTPS_PROXY'] = "http://127.0.0.1:10809"
                openai.api_key = self.api_key_list
                openai.api_base = "https://api.xiaoai.plus/v1"
                request_text = sensitive(request_text)
                output = openai.Embedding.create(
                    model=self.model_name,
                    input=[request_text],
                    temperature=self.temperature
                )
                break
            except Exception as e:
                print('Exception:', e)
                sys.exit()
        return output['data'][0]["embedding"]

    def send_request_entity_pair(self, sys_prompt, user_message):
        """
        """
        messages = []
        sys_prompt = sensitive(sys_prompt)
        zero_shot_prompt_message = {'role': 'system', 'content': sys_prompt}
        messages.append(zero_shot_prompt_message)
        # user_message = f"{sys_prompt}\n{user_message}"
        user_message = sensitive(user_message)
        message = {"role": "user", "content": user_message}
        print(f"LLM的Prompt是{'*' * 100}\n{message['content']}")
        messages.append(message)
        while True:
            try:
                os.environ['HTTPS_PROXY'] = "http://127.0.0.1:10809"
                openai.api_key = self.api_key_list
                openai.api_base = "https://api.xiaoai.plus/v1"
                output = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature
                )
                output.choices[0].message.content = revers_sensitive(output.choices[0].message.content)
                answer = output.choices[0].message.content
                answer = revers_sensitive(answer)
                print(answer)
                return answer
            except Exception as e:
                print('Exception:', e)
                print("原始Prompt：")
                sys.exit()
    def forward(self, sys_prompt, user_message, question_type='entity_pair')->list:
        """
        """
        output = []
        if question_type in ['entity_pair', 'entity_merge']:
            output = self.send_request_entity_pair(sys_prompt, user_message)
            # output = self.send_request_turbo_chat_k_shot(prompt, share_content, question, examples)
        elif question_type == 'encoder':
            output = self.send_request_encoder(user_message)

        # print(output)
        return output
    
    def postprocess(self, output):
        """
        """
        model_output = None
        try:

            if "gpt" in self.model_name:
                model_output = output['choices'][0]['message']['content']

            elif self.model_name == 'text-davinci-003':
                model_output = output['choices'][0]['text']

            if not model_output:
                print("Warning: Empty Output ")
        except Exception as e:
            print('Exception:', e)
            model_output = '【解析】\n<eoe>\n【答案】'
            print("Warning error: Empty Output ")
        return model_output

    def __call__(self, sys_prompt:str, user_message:str, question_type:str):
        return self.forward(sys_prompt, user_message, question_type)


def sensitive(sentence):
    sentence = sentence.replace("阴道", "term-YD")
    sentence = sentence.replace("射精", "term-SJ")
    return sentence

def revers_sensitive(sentence):
    sentence = sentence.replace("term-YD", "阴道")
    sentence = sentence.replace("term-SJ", "射精")
    return sentence

    
