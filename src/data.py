import re
import os
import json
import math
import random
import datasets
from tqdm import tqdm
from functools import partial
from glob import glob
from contextlib import nullcontext
from transformers.utils import logging
from src import apply_chat_template, add_eos, split_file_dir_name_ext

logger = logging.get_logger(__name__)


# RETRIEVAL_CAND = [(1024,1), (512,2), (256,4), (128,8), (512,1), (256,2), (128,4)]
RETRIEVAL_CAND = [(1024,1)]


class Data:
    def _process_language_modeling(data, indices, tokenizer, min_length, max_length):
        outputs = {'input_ids': [], 'attention_mask': [], "labels": [], "length": [], "index": [],'query':[],'query_start':[]}

        for i, text in enumerate(data['text']):
            # truncate text for faster processing
            encoded = tokenizer(text)
            if len(encoded["input_ids"]) < min_length:
                continue
            elif len(encoded['input_ids']) < max_length:
                encoded = add_eos(encoded, tokenizer.eos_token_id)
            else:
                for k, v in encoded.items():
                    encoded[k] = v[:max_length]

            encoded["labels"] = encoded["input_ids"].copy()
            encoded['query'] =  []
            encoded['query_start'] = [0]
            for k, v in encoded.items():
                outputs[k].append(v)
            # length is required for grouping
            outputs["length"].append(len(encoded['input_ids']))
            outputs["index"].append(indices[i])
            # outputs['query'] = [0]

        return outputs

    def _process_instruction_tuning(data, indices, tokenizer, chat_template, min_length, max_length, eval_mode=False):
        outputs = {'input_ids': [], 'attention_mask': [], "labels": [], "length": [], "index": [],'query':[],'query_start':[]}
        for i, source in enumerate(data['conversations']):
            if source[0]["role"] != 'user':
                # Skip the first one if it is not from user
                source = source[1:]

            # NOTE: in evaluation, we only use the first turn in the conversation
            if eval_mode:
                # a string (the expected output from the assistant)
                if len(source) > 1:
                    labels = source[1]['content']
                else:
                    labels = None
                source = source[:1]
            # Query = "Next I will give you one or more questions that will guide you to extract key information from the following document to answer more of the corresponding questions."
            # for idx,message in enumerate(source):
            #     if message['role'] == "user":
            #         #if idx==0:
            #         content = message['content'].split("\n")[-1]
            #         if "Question: " in content:
            #             content = content.split("Question: ")[-1]
            #         Query = f"{Query} Question {int(idx/2)+1}: {content} "
            #         #else:
            #         #    content = message['content'].split("\n")[-1]
            #         #    Query = f"{Query} Question {int(idx/2)+1}: {content}"
            # Query = Query + " The question given is the end. \n\n"
            # print(Query)
        # if message['role'] == "user":
            # message["content"]= "Please compress the important information in the query below\nQuery: " + message['content'].split("\n")[-1] +"\n"+  message['content']
            # Query = "Please compress the important information in the query below\nQuery:"
            # for idx in range(len(source)):
            #     if source[idx]["role"] == 'user' and idx !=0:
            #         Query = Query + "Question:"
            encoded = apply_chat_template(
                chat_template,
                source,
                tokenizer=tokenizer,
                # only return labels in evaluation mode
                return_labels=not eval_mode,
                add_generation_prompt=eval_mode,
                Query=None
            ).encoded
            #用了模板之后，这里的返回是已经insertquery的了
            """
            query : input_query
            query_length
            query_start : start : start + len(query) query_length其实不需要
            
            """



            # skip data that not fall in between min_length and max_length
            if min_length is not None and len(encoded["input_ids"]) < min_length:
                continue
            if "query_length" in encoded:
                if max_length is not None and len(encoded["input_ids"]) > max_length + len(encoded["query"]):
                    continue
            else:
                if max_length is not None and len(encoded["input_ids"]) > max_length:
                    continue

            if eval_mode:
                encoded["labels"] = labels

            for k, v in encoded.items():
                outputs[k].append(v)
            outputs['length'].append(len(encoded['input_ids']))
            outputs['index'].append(indices[i])

            # def find_first_non_minus_100_index(lst):
            #     for index, value in enumerate(lst):
            #         if value != -100:
            #             return index
            #     return -1  # 如果没有找到符合条件的元素，返回 -1
            # real_label_length =len(encoded["labels"]) - find_first_non_minus_100_index(encoded["labels"])
            # math_length = len(encoded["labels"]) % 1024
            # if real_label_length > math_length:
            #     # print("---")
            #     with open("static.txt",'a+',encoding='utf-8') as f:
            #         length = len(encoded["labels"])
            #         a = f"总长度为{length},真实标签长度为:{real_label_length},被截断的标签长度为{real_label_length-math_length}\n"
            #         f.write(a)
            #         f.close()
                # print(real_label_length-math_length)
        #TODO
        #看labels中第一个非-100的index，即为对应的标签长度，然后得到一个labels真实长度
        #用全部长度取余1024，用真实长度 - 取余结果，可以得到labels被mask掉的长度
        return outputs

    def prepare_train_data(data_files=None, tokenizer=None, max_length=4096, min_length=512, chat_template="vicuna", seed=42, cache_dir=None, load_from_cache_file=None):
        if data_files is None:
            return None

        if isinstance(data_files, list):
            logger.info(f"Loading training data from {data_files}...")
        elif isinstance(data_files, str):
            logger.info(f"Loading training data from {data_files}...")
            data_files = [data_files]
        else:
            raise ValueError(f"Invalid training data {data_files}!")

        data_2_num_sample = {}

        for data_file in data_files:
            match = re.search("\[(\d*)\]", data_file)
            if match:
                max_sample_num = int(match.group(1))
                data_file = re.sub("\[(\d*)\]", "", data_file)
            else:
                max_sample_num = None
            print(data_file)
            data_2_num_sample[data_file] = max_sample_num

        random.seed(seed)

        train_datasets = []
        count=0
        path = "/data/chenss/train_data_all_.json"
        if os.path.exists(path):
            dataset = datasets.load_dataset('json', data_files=path, split='train')# dataset_dict = DatasetDict()
        else:
            for data_file, max_sample_num in data_2_num_sample.items():
                if os.path.isdir(data_file) and os.path.exists(os.path.join(data_file, "dataset_info.json")):
                    # the dataset may be save_to_disk in advance
                    dataset = datasets.load_from_disk(data_file)
                else:
                    # the dataset is a json file
                    dataset = datasets.load_dataset('json', data_files=data_file, split='train', cache_dir=cache_dir)
                    column_names = dataset.column_names
                    if "text" in column_names:
                        process_fn = partial(
                            Data._process_language_modeling,
                            tokenizer=tokenizer,
                            min_length=min_length,
                            max_length=max_length
                        )
                    elif "conversations" in column_names:
                        process_fn = partial(
                            Data._process_instruction_tuning,
                            tokenizer=tokenizer,
                            chat_template=chat_template,
                            min_length=min_length,
                            max_length=max_length
                        )
                    else:
                        raise ValueError(f"Found neither 'text' nor 'conversations' in the training data!")

                    dataset = dataset.map(process_fn, batched=True, num_proc=32, remove_columns=dataset.column_names, batch_size=32, with_indices=True, load_from_cache_file=False)#load_from_cache_file)
                    # print(dataset)
                    # path = "/".join(data_file.split("/")[:-1]) +"/" +data_file.split("/")[-1].split(".json")[0]+"_nq.json"
                    # dataset.to_json(path)
                    # # dataset = datasets.load_dataset('json', data_files=data_file, split='train')# dataset_dict = DatasetDict()
                    # datasets.Dataset.from_dict(outputs)
                    # output_file = "/data/chenss/UltraGist/main/cache/" + data_file.split("/")[-1]
                    # with open(output_file, 'w') as f:
                        # json.dump(dataset, f)

                if max_sample_num is not None and len(dataset) > max_sample_num:
                    dataset = dataset.train_test_split(max_sample_num, seed=seed)["test"]

                # index column is useless in training
                if "index" in dataset.column_names:
                    dataset = dataset.remove_columns(["index"])

                # path = "/".join(data_file.split("/")[:-1]) +"/" +data_file.split("/")[-1].split(".json")[0]+"_new_.json"
                # # dataset.to_json(path)
                # print(data_file)
                # dataset = datasets.load_dataset('json', data_files=data_file, split='train') # dataset_dict = DatasetDict()
                # if "query_one_detail_book.train.8K_new_.json" in data_file or  "query_one_detail_paper.train.8K_new_.json" in data_file:
                #     max_num = 20000
                #     dataset = dataset.train_test_split(max_num, seed=seed)["test"]
                print(dataset)

                train_datasets.append(dataset)

            dataset = datasets.concatenate_datasets(train_datasets)
            # dataset.to_json("/data/chenss/train_data_all.json")
            print(dataset)
        # path = "train_data.json"
        # dataset = datasets.load_dataset('json', data_files=path, split='train')# dataset_dict = DatasetDict()
        return dataset

    def prepare_eval_data(data_files=None, tokenizer=None, max_length=4096, min_length=512, chat_template="vicuna", max_eval_num=None, cache_dir=None, seed=42, load_from_cache_file=None):
        if data_files is None:
            return None

        random.seed(seed)

        if max_eval_num is not None:
            dataset = datasets.load_dataset('json', data_files=data_files, split=f'train[:{max_eval_num}]', cache_dir=cache_dir)
        else:
            dataset = datasets.load_dataset('json', data_files=data_files, split='train', cache_dir=cache_dir)

        column_names = dataset.column_names
        if "text" in column_names:
            process_fn = partial(
                Data._process_language_modeling,
                tokenizer=tokenizer,
                min_length=min_length,
                max_length=max_length
            )
        elif "conversations" in column_names:
            process_fn = partial(
                Data._process_instruction_tuning,
                tokenizer=tokenizer,
                chat_template=chat_template,
                min_length=min_length,
                max_length=max_length,
                eval_mode=True,
            )
        else:
            raise ValueError(f"Found neither 'text' nor 'conversations' in the training data!")

        dataset = dataset.map(process_fn, batched=True, num_proc=2, remove_columns=dataset.column_names, with_indices=True, load_from_cache_file=load_from_cache_file)
        return dataset