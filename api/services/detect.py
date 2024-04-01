import base64
import copy
import json
import os
from io import BytesIO

import requests
import torch
import time

from PIL import Image
from linker_atom.lib.log import logger

from api.model_protector import ModelProtector
from api.utils import LRUCache, get_model_state_dict
from api.config import EnvVar
from api.schemas.response import Answer
from llava.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token, KeywordsStoppingCriteria
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

dectypt = os.getenv('IS_ENCRYPT') != 'false'


class Engine:
    def __init__(self) -> None:
        self.model = LRUCache(1)
        self.base_model = LRUCache(2)

    def load_model(self, model_id, resources_prefix="resources"):
        self.resources_prefix = resources_prefix
        self.addapter_resources_prefix = resources_prefix
        if self.model.has(model_id):
            return self.model.get(model_id)
        else:
            self.model_id = model_id

            if dectypt:
                self.base_model_dectypted = self.get_base_model_id(model_id)
                if self.base_model_dectypted:
                    self.base_model_id = self.base_model_dectypted
                    self.lora_model_id = None
                else:
                    a = ModelProtector(xor_key=12, user_id="omchat", model_version=1)
                    encrypt_model_path = os.path.join("{}/{}".format(self.resources_prefix, model_id + '.linker'))
                    if not os.path.exists(encrypt_model_path):
                        error_info = f"{encrypt_model_path} not exists"
                        raise Exception(error_info)
                    try:
                        model_path, out_path = a.decrypt_model(encrypt_model_path)
                        self.addapter_resources_prefix = out_path
                        self.lora_model_id = model_id
                        self.base_model_id = None
                    except:
                        a.remove_model(out_path)
                        raise Exception("解密失败！")
            else:
                if self._is_base_model(model_id):
                    self.lora_model_id = None
                    self.base_model_id = model_id
                else:
                    self.lora_model_id = model_id
                    self.base_model_id = None

            if self.base_model_id:
                base_model = self._load_base_model(self.base_model_id)
                model = self._load_model_adapter(base_model, self.lora_model_id)
                self.model.put(self.base_model_id, model)
            else:
                base_model_id = self._get_base_model_id(self.lora_model_id)
                base_model = self._load_base_model(base_model_id)
                model = self._load_model_adapter(base_model, self.lora_model_id)

            if dectypt and self.lora_model_id is not None:
                a.remove_model(out_path)
            return model

    def _is_base_model(self, model_id):
        adapter_config_path = os.path.join("{}/{}".format(self.addapter_resources_prefix, model_id),
                                           "adapter_config.json")
        if os.path.isfile(adapter_config_path):
            return False
        return True

    def get_base_model_id(self, model_id):
        if os.path.isfile("{}/{}.tar.gz".format(self.resources_prefix, model_id)):
            return model_id
        return None

    def _get_base_model_id(self, model_id):
        with open(os.path.join("{}/{}".format(self.addapter_resources_prefix, model_id), "adapter_config.json"),
                  "r") as f:
            adapter_config = json.load(f)
        base_model_id = adapter_config.get("base_model_name_or_path", None)
        if "/" in base_model_id:
            base_model_id = base_model_id.split("/")[-1]
        assert base_model_id is not None, "adapter config has not 'base_model_name_or_path'"
        return base_model_id

    def _load_model_adapter(self, base_model, model_id):
        model = base_model[0].mllm_engine.driver_worker.model_runner.model
        base_state_dict = base_model[1]
        model_id_path = None
        if model_id:
            model_id_path = os.path.join("{}/{}".format(self.addapter_resources_prefix, model_id))
        model.load_lora_weights(model_id_path, base_state_dict)
        model.to(torch.float16)

        self.model.put(model_id, base_model[0])
        return base_model[0]

    def _load_base_model(self, base_model_id):
        if self.base_model.has(base_model_id):
            return self.base_model.get(base_model_id)
        if dectypt:
            status = os.system(
                "openssl aes-256-cbc -d -salt -k HZlh@2023 -in {}/{}.tar.gz | tar -xz -C {}/".format(
                    self.resources_prefix,
                    base_model_id, self.resources_prefix))
            if status != 0:
                raise RuntimeError("unzip failed, error code is {}. please connect engineer".format(status))
        base_model = self.base_model.get_last_model()
        if base_model:
            model = base_model.mllm_engine.driver_worker.model_runner.model

            model.load_weights(model_name_or_path="{}/{}".format(self.resources_prefix, base_model_id)
                               )
            model.cuda()
        else:
            base_model = MLLM(
                model="{}/{}".format(self.resources_prefix, base_model_id),
                tokenizer="{}/{}".format(self.resources_prefix, base_model_id),
                gpu_memory_utilization=EnvVar.GPU_MEMORY_UTILIZATION,
                dtype="float16",
                lora_weight_id="{}/{}".format(self.addapter_resources_prefix, self.model_id),
                max_num_batched_tokens=EnvVar.MAX_NUM_BATCHED_TOKENS,
                enforce_eager=False
            )

        base_state_dict = {}
        for name, para in get_model_state_dict(base_model).items():
            base_state_dict[name] = copy.deepcopy(para).to("cpu")
        self.base_model.put(base_model_id, (base_model, base_state_dict))
        del base_model
        if dectypt:
            os.system("rm -rf {}/{}".format(self.resources_prefix, base_model_id))
        return self.base_model.get(base_model_id)

    def load_one_image(self, image, src_type):
        # time.sleep(5)
        # print("_load_one_image")
        if not image:
            return None
        image_file = image
        if src_type == "url":
            response = requests.get(image_file, timeout=10)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        elif src_type == "local":
            image = Image.open(image_file).convert('RGB')
        elif src_type == "base64":
            image = Image.open(BytesIO(base64.b64decode(image_file))).convert('RGB')
        elif src_type == "mmap":
            image = mmap_to_pil(value=image_file)
        else:
            assert 0, "src_type is not true"
        return image
        image_tensor = \
            self.mllm_engine.driver_worker.model_runner.model.model.image_processor(image, return_tensors='pt')[
                'pixel_values'][0]
        return image_tensor.half().cuda()

    def load_model_llava(self,model_id):
        if self.model.has(model_id):
            return self.model.get(model_id)
        model_path = os.path.expanduser(model_id)
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base=None,
                                                                               model_name=model_name)
        self.model.put(model_id, (tokenizer, model, image_processor, context_len))
        return tokenizer, model, image_processor, context_len

    async def batch_predict(
            self,
            model_id,
            prompts,
            initial_prompt,
            temperature=1,
            max_tokens=1024,
            top_p=1,
    ):
        disable_torch_init()
        model_path = os.path.expanduser(model_id)
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = self.load_model_llava(model_id)

        if 'llama-2' in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"
        conv_mode = os.getenv("chat_format","mpt")
        generated_texts = []
        for i, prompt in enumerate(prompts):
            question = prompt.records[0].user
            image = prompt.image if prompt.image else None
            src_type = prompt.src_type if prompt.src_type else  None

            if image:
                image = self.load_one_image(image, src_type)
                image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                images = image_tensor.unsqueeze(0).half().cuda()

                if getattr(model.config, 'mm_use_im_start_end', False):
                    qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
                else:
                    qs = DEFAULT_IMAGE_TOKEN + '\n' + question
            else:
                images = None
                qs = question


            conv = conv_templates[conv_mode].copy()
            if initial_prompt:
                conv.system=initial_prompt
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(
                0).cuda()

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = []
            stopping_criteria = [
                KeywordsStoppingCriteria(keywords, tokenizer, input_ids)] if conv.version == "v0" else None

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=images,
                    do_sample=True if temperature > 0 else False,
                    temperature=temperature,
                    max_new_tokens=max_tokens,
                    use_cache=True,
                    stopping_criteria=stopping_criteria,
                )

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()




            text = outputs
            if image is not None:
                if images.shape[-1] == 336:
                    input_tokens = input_token_len + 576
                else:
                    input_tokens = input_token_len + 256
            else:
                input_tokens = input_token_len
            print(len(output_ids))
            output_tokens = output_ids.shape[1]-input_token_len
            mean_prob = None
            probs = {}
            # total_prob = math.exp(output.outputs[0].cumulative_logprob)
            # mean_prob = total_prob**(1/output_tokens)
            # probs={}
            # for name, logprob in output.outputs[0].logprobs:
            #     probs[name] = torch.exp_(torch.tensor(logprob,dtype=torch.float)).item()
            generated_texts.append(
                Answer(content=text, input_tokens=input_tokens, output_tokens=output_tokens, mean_prob=mean_prob,
                       probs=probs))
            logger.info(text)
        logger.info(generated_texts)
        return generated_texts

    async def profromance_banchmark(
            self,
            model_id,
            prompts,
            initial_prompt,
            temperature=1,
            max_tokens=1024,
            top_p=1,
            fixed_length=(None,None)
    ):
        disable_torch_init()
        model_path = os.path.expanduser(model_id)
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = self.load_model_llava(model_id)
        input_tokens_number, output_tokens_number = fixed_length

        if 'llama-2' in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"
        conv_mode = os.getenv("chat_format","mpt")
        generated_texts = []
        for i, prompt in enumerate(prompts):
            question = prompt.records[0].user
            image = prompt.image if prompt.image else None
            src_type = prompt.src_type if prompt.src_type else  None

            if image:
                image = self.load_one_image(image, src_type)
                image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                images = image_tensor.unsqueeze(0).half().cuda()

                if getattr(model.config, 'mm_use_im_start_end', False):
                    qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
                else:
                    qs = DEFAULT_IMAGE_TOKEN + '\n' + question
            else:
                images = None
                qs = question


            conv = conv_templates[conv_mode].copy()
            if initial_prompt:
                conv.system=initial_prompt
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(
                0).cuda()
            image_height = model.model.vision_tower.image_processor.crop_size['height']
            image_token_len = (image_height/14)**2
            if input_tokens_number is not None:
                assert not (image and input_tokens_number < image_token_len + 50), \
                    f"please input tokens lenght , at least more than {image_token_len + 50}"
                if image:
                    origin_input_token_len = input_ids.shape[-1]
                    if input_tokens_number < origin_input_token_len:
                            input_ids = input_ids[:, :(input_tokens_number-image_token_len)]
                    else:
                        input_ids = torch.nn.functional.pad(input_ids,
                                                            (0, int(input_tokens_number - origin_input_token_len-image_token_len)),"constant",6000)
                    logger.warning(
                        f"profromance banchmark input tokens {origin_input_token_len} -> {input_ids.shape[-1]}")

                else:
                    origin_input_token_len = input_ids.shape[-1]
                    if input_tokens_number < origin_input_token_len:
                        input_ids = input_ids[:,:input_tokens_number]
                    else:
                        input_ids = torch.nn.functional.pad(input_ids, (0,input_tokens_number - origin_input_token_len))
                    logger.warning(
                        f"profromance banchmark input tokens {origin_input_token_len} -> {input_ids.shape[-1]}")

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = []
            stopping_criteria = [
                KeywordsStoppingCriteria(keywords, tokenizer, input_ids)] if conv.version == "v0" else None

            if output_tokens_number:
                max_tokens = output_tokens_number
            model.generation_config.eos_token_id = None
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    generation_config=model.generation_config,
                    images=images,
                    do_sample=True if temperature > 0 else False,
                    temperature=temperature,
                    max_new_tokens=max_tokens,
                    use_cache=True,
                    stopping_criteria=stopping_criteria,
                )

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()




            text = outputs
            if image is not None:
                if images.shape[-1] == 336:
                    input_tokens = input_token_len + 576
                else:
                    input_tokens = input_token_len + 256
            else:
                input_tokens = input_token_len
            print(len(output_ids))
            output_tokens = output_ids.shape[1]-input_token_len
            mean_prob = None
            probs = {}
            # total_prob = math.exp(output.outputs[0].cumulative_logprob)
            # mean_prob = total_prob**(1/output_tokens)
            # probs={}
            # for name, logprob in output.outputs[0].logprobs:
            #     probs[name] = torch.exp_(torch.tensor(logprob,dtype=torch.float)).item()
            generated_texts.append(
                Answer(content=text, input_tokens=input_tokens, output_tokens=output_tokens, mean_prob=mean_prob,
                       probs=probs))
            logger.info(text)
        logger.info(generated_texts)
        return generated_texts

if __name__ == "__main__":
    e = Engine()
    s_t = time.time()
    model = e.load_model(model_id="omchat-llava-qllama-7b-chat-v1-1-finetune_qlora_zh_n67",
                         # "lq_mcqa_0_314",#"omchat-llava-qllama-7b-chat-v1-1-qllama-finetune_zh_n97",#"omchat-llava-vicuna-7b-v1.5-v1-1-finetune_zh_n92",",#
                         resources_prefix="../../../llm_models"
                         )
    print(time.time() - s_t)

    sampling_params = SamplingParams(
        temperature=0.9, max_tokens=512, top_p=1.0, stop=["<|im_end|>"]
    )
    images = []
    texts = []

    res = model.generate(
        prompts=[[{"user": "图片上有什么"}]],
        images=[{"src_type": "url",
                 "image_src": "https://img0.baidu.com/it/u=56109659,3345510515&fm=253&fmt=auto&app=138&f=JPEG?w=889&h=500"}],
        choices=[[]],
        sampling_params=sampling_params,
        initial_prompt="你好",
    )
    generated_texts = []
    for output in res:
        text = output.outputs[0].text
        input_tokens = len(output.prompt_token_ids)
        output_tokens = len(output.outputs[0].token_ids)
        generated_texts.append(Answer(content=text, input_tokens=input_tokens, output_tokens=output_tokens))
        print(output.prompt)
    print(generated_texts)
    print(time.time() - s_t)
    print("done")
