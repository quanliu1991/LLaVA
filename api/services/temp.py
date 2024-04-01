#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author   : liuquan
# @date     : 2023/10/18:5:44 PM
import os
import re
os.environ["CUDA_VISIBLE_DEVICES"]="2"
from vllm.model_executor.models.llava import transpose

fan_in_fan_out=False
scaling= 0.25


import torch

# def _is_lora_name_match(lora_name, lora_weight_name):
#     if lora_name in lora_weight_name:
#         return True
#     else:
#         return False
#
# def _get_delta_weight( name) -> torch.Tensor:
#     lora_A_name = name.replace("weight", "lora_A.weight")
#     lora_B_name = name.replace("weight", "lora_B.weight")
#     return (
#             transpose(
#                 lora_weight_dict[lora_B_name].cuda() @ lora_weight_dict[lora_A_name].cuda(),
#                 fan_in_fan_out
#             )
#             * scaling
#     )
#
# lora=torch.load("/app/llm_models/omchat-llava-qllama-7b-chat-v1-1-finetune_qlora_zh_n67/adapter_model.bin")
#
# lora_weight_dict = {}
# datle_weight_dict = {}
# lora_weight_name =[]
# for lora_name,loaded_weight in lora.items():
#     if "layers.0" in lora_name:
#         profix_name = lora_name.split(".layers.")[0]
#         lora_name = lora_name.replace(profix_name, "model.llama_model")
#         lora_weight_dict[lora_name] = loaded_weight
#         name = re.sub(r"lora_*..", "", lora_name)
#         if _is_lora_name_match(name,lora_weight_name):
#             datle_weight_dict[name]= loaded_weight = _get_delta_weight(name)
#         else:
#             lora_weight_name.append(name)
#             continue





import pickle
with open("test_model_all.pkl","rb") as f:
    all_model = pickle.load(f)

with open("lora_and_base.pkl","rb") as f:
    lora_load_result = pickle.load(f)

# all_layer_0={}
# for i in all_model.keys():
#     if "layers.0" in i:
#         all_layer_0[i]=all_model[i]
#
# del all_model
# torch.cuda.empty_cache()
#
# lora_layer_0={}
# for i in lora_load_result.keys():
#     if "layers.0" in i:
#         lora_layer_0[i]=lora_load_result[i]
#
# del lora_load_result
# torch.cuda.empty_cache()

for name in all_model.keys():
    try:
        a= all_model[name]-lora_load_result[name]
    except:
        print(name)
        continue
    print(name,a)

for name in all_model.keys():
    try:
        a= all_model[name]-lora_load_result[name]
    except:
        print(name)
        continue
    # nonzero_indices = torch.nonzero(a)
    # nonzero_count = nonzero_indices.size(0)
    num = (a > 0.00005).sum()
    print(name,num/a.numel())
    print(name,num)
    # print(name,nonzero_count/a.numel())



print("a")

a={"model_id": "omchat-llava-vicuna-7b-v1.5-v1-1-finetune_zh_n92", "prompts": [{"image": "https://studio.linker.cc/webdav/om_studio/dataSet/64707d85e4b0e7672b852ab8.jpg", "src_type": "url", "records": [{"user": "以商店店长的身份，描述图中需要注意的内容和细节，并给出合理建议。"}]}, {"image": "https://studio.linker.cc/webdav/om_studio/dataSet/64707d29e4b0e7672b84f2ba.jpg", "src_type": "url", "records": [{"user": "以商店店长的身份，描述图中需要注意的内容和细节，并给出合理建议。"}]}, {"image": "https://studio.linker.cc/webdav/om_studio/dataSet/64707cace4b0e7672b84ee25.jpg", "src_type": "url", "records": [{"user": "以商店店长的身份，描述图中需要注意的内容和细节，并给出合理建议。"}]}, {"image": "https://studio.linker.cc/webdav/om_studio/dataSet/64707b39e4b0e7672b84ca3d.jpg", "src_type": "url", "records": [{"user": "以商店店长的身份，描述图中需要注意的内容和细节，并给出合理建议。"}]}, {"image": "https://studio.linker.cc/webdav/om_studio/dataSet/64707d39e4b0e7672b84f354.jpg", "src_type": "url", "records": [{"user": "以商店店长的身份，描述图中需要注意的内容和细节，并给出合理建议。"}]}, {"image": "https://studio.linker.cc/webdav/om_studio/dataSet/64707ca8e4b0e7672b84ee09.jpg", "src_type": "url", "records": [{"user": "以商店店长的身份，描述图中需要注意的内容和细节，并给出合理建议。"}]}, {"image": "https://studio.linker.cc/webdav/om_studio/dataSet/64707d34e4b0e7672b84f323.jpg", "src_type": "url", "records": [{"user": "以商店店长的身份，描述图中需要注意的内容和细节，并给出合理建议。"}]}, {"image": "https://studio.linker.cc/webdav/om_studio/dataSet/64707d88e4b0e7672b852af2.jpg", "src_type": "url", "records": [{"user": "以商店店长的身份，描述图中需要注意的内容和细节，并给出合理建议。"}]}, {"image": "https://studio.linker.cc/webdav/om_studio/dataSet/64707d8ee4b0e7672b852b52.jpg", "src_type": "url", "records": [{"user": "以商店店长的身份，描述图中需要注意的内容和细节，并给出合理建议。"}]}, {"image": "https://studio.linker.cc/webdav/om_studio/dataSet/64707b53e4b0e7672b84ce7e.jpg", "src_type": "url", "records": [{"user": "以商店店长的身份，描述图中需要注意的内容和细节，并给出合理建议。"}]}], "initial_prompt": "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."}