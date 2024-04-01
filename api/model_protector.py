#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author   : liuquan
# @date     : 2023/10/19:4:04 PM
import tarfile
import os
import uuid
import shutil
import base64
from time import time
from random import randint


class ModelProtector:
    def __init__(self, xor_key: int, user_id: str, model_version: int = 0):
        self.xor_key = xor_key
        self.user_id = user_id
        self.model_version = model_version

    def xore(self, data, key):
        result = []

        for a in data:
            if a == 0 or a == key:
                result.append(a)
            else:
                result.append(a ^ key)
        return bytes(result)

    def confuse(self, data, key):
        s = time()
        if len(data) > 10240:
            head = self.xore(data[:10240], key)
            c_data = b''.join([head, data[10240:]])
        else:
            c_data = self.xore(data, key)
        print('conf:{}'.format(time() - s))
        return c_data

    def get_header_legacy(self, model_length):
        LH_mark = bytes('LH', encoding='utf-8')
        model_length = int(model_length).to_bytes(length=8, byteorder='big', signed=False)
        version_num = int(self.model_version).to_bytes(length=1, byteorder='big', signed=False)
        customer_id = base64.b85encode(self.user_id.encode('utf-8'))
        head_length = int(len(LH_mark + model_length + version_num + customer_id) + 2).to_bytes(length=2,
                                                                                                byteorder='big',
                                                                                                signed=False)
        return LH_mark + head_length + version_num + model_length + customer_id

    def get_header(self, model_length):
        LH_mark = bytes('HL', encoding='utf-8')
        model_length = int(model_length).to_bytes(length=8, byteorder='big', signed=False)
        version_num = int(self.model_version).to_bytes(length=1, byteorder='big', signed=False)
        customer_id = base64.b85encode(self.user_id.encode('utf-8'))
        head_length = int(len(LH_mark + model_length + version_num + customer_id) + 2).to_bytes(length=2,
                                                                                                byteorder='big',
                                                                                                signed=False)
        return LH_mark + head_length + version_num + model_length + customer_id

    def decode_header(self, file):
        mark = str(file[0:2], "utf-8")
        head_length = int().from_bytes(file[2:4], byteorder='big', signed=False)
        head_info = file[:head_length]
        model_content = file[head_length:]

        version_info = int().from_bytes(head_info[4:5], byteorder='big', signed=False)

        model_length = int().from_bytes(head_info[5:13], byteorder='big', signed=False)

        customer_id = str(base64.b85decode(head_info[13:]), "utf-8")

        if self.user_id != customer_id:
            raise Exception('Customer Id Mismatch!!!!!!')

        if model_length != len(model_content):
            raise Exception('Model Length Mismatch!!!!!!')

        return mark, model_content

    def encrypt_model_legacy(self, source_dir, out_path='.'):
        print('------------------encrypting model----------------')
        s_time = time()
        tar = tarfile.open(os.path.join(out_path, os.path.basename(source_dir) + "_temp.tar"), "x")
        tar.add(source_dir, arcname=os.path.basename(source_dir))
        tar.close()

        with open(os.path.join(out_path, os.path.basename(source_dir) + "_temp.tar"), 'rb') as f:
            encrypted = self.xore(f.read(), self.xor_key)

        os.remove(os.path.join(out_path, os.path.basename(source_dir) + "_temp.tar"))
        header = self.get_header_legacy(len(encrypted))

        with open(os.path.join(out_path, os.path.basename(source_dir) + '.linker'), 'wb') as f:
            f.write(header + encrypted)

        print('Encrypt time: {}'.format(time() - s_time))

        return os.path.join(out_path, os.path.basename(source_dir) + '.linker')

    def encrypt_model(self, source_dir, out_path='.'):
        print('------------------encrypting model----------------')
        s_time = time()
        tar = tarfile.open(os.path.join(out_path, os.path.basename(source_dir) + "_temp.tar"), "x")
        tar.add(source_dir, arcname=os.path.basename(source_dir))
        tar.close()

        with open(os.path.join(out_path, os.path.basename(source_dir) + "_temp.tar"), 'rb') as f:
            encrypted = self.confuse(f.read(), self.xor_key)

        os.remove(os.path.join(out_path, os.path.basename(source_dir) + "_temp.tar"))
        header = self.get_header(len(encrypted))

        with open(os.path.join(out_path, os.path.basename(source_dir) + '.linker'), 'wb') as f:
            f.write(header + encrypted)

        print('Encrypt time: {}'.format(time() - s_time))

        return os.path.join(out_path, os.path.basename(source_dir) + '.linker')

    def decrypt_model(self, source_dir, out_path='/tmp'):
        print('------------------decrypting model----------------')
        s_time = time()
        with open(source_dir, 'rb') as f:
            file = f.read()
            mark, model_content = self.decode_header(file)

            if mark == 'HL':
                decrypted = self.confuse(model_content, self.xor_key)
            elif mark == 'LH':
                decrypted = self.xore(model_content, self.xor_key)

        out_path = os.path.join(out_path, str(uuid.uuid4()))

        if not os.path.exists(out_path):
            os.makedirs(out_path)

        with open(os.path.join(out_path, os.path.basename(source_dir) + ".tar"), 'wb') as f:
            f.write(decrypted)
        try:
            with tarfile.open(os.path.join(out_path, os.path.basename(source_dir) + ".tar")) as f:
                f.extractall(path=out_path)
        except tarfile.ReadError:
            raise RuntimeError('Invalid model file')

        dirs = [i for i in os.listdir(out_path) if os.path.isdir(out_path + '/' + i)]

        print('Decrypt time: {}'.format(time() - s_time))

        return os.path.join(out_path, dirs[0]) if dirs else out_path, out_path

    def remove_model(self, source_dir):
        shutil.rmtree(source_dir)
        # print('/'.join(source_dir.split('/')[:-1]))
        # shutil.rmtree('/'.join(source_dir.split('/')[:-1]))


if __name__ == '__main__':


    a = ModelProtector(xor_key=12, user_id='omchat', model_version=1)
    model_path, out_path = a.decrypt_model('/app/omchat/resources/lq_mcqa_0_314.linker',out_path="/app/omchat/resources/")
    # a.encrypt_model('/app/omchat/resources/omchat_mcqa_1_2519.linker')


