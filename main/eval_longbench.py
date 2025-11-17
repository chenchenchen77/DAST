import os
import datasets
import json
import torch
from tqdm import tqdm
from typing import Optional, Dict, List
from functools import partial
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from accelerate import Accelerator
from transformers import HfArgumentParser
from transformers.utils import logging
from torch.utils.data import DataLoader

from src import ModelArgs, DefaultDataCollator, FileLogger, get_model_and_tokenizer, makedirs, apply_chat_template
from .longbench_utils import DATASET2PROMPT, DATASET2MAXNEWTOKENS, DATASET2CATEGORY, scorer

logger = logging.get_logger(__name__)


@dataclass
class Args(ModelArgs):
    eval_data: str = field(
        default="ultragist:longbench/narrativeqa.jsonl",# test.json",
        metadata={'help': 'The evaluation json data path.'}
    )
    output_dir: str = field(
        default="data/results/longbench/",
        metadata={'help': 'The base directory for saving results and logs.'}
    )
    result_dir: Optional[str] = field(
        default=None,
        metadata={'help': 'The directory relative to output_dir for saving results.'}
    )

    dataset_names: List[str] = field(
        default_factory=lambda: ['narrativeqa'],
        metadata={'help': 'Which dataset to evaluate?'}
    )

    max_length: int = field(
        default=31500,
        metadata={'help': 'Max input length.'}
    )
    truncate_from_middle: bool = field(
        default=True,
        metadata={'help': 'Truncate inputs from the middle.'}
    )
    load_result: bool = field(
        default=False,
        metadata={'help': 'Load result from saved files?'}
    )

    do_sample: bool = False


def process_longbench(data, indices, tokenizer, chat_template, prompt_templates:Optional[Dict]=None, max_length_global=3500, truncate_from_middle=True):
    outputs = {'input_ids': [], 'attention_mask': [], "dataset": [], "index": [],'query':[],"query_length":[],"query_start":[]}

    for input, context, dataset, index in zip(data['input'], data['context'], data['dataset'], indices):
        if dataset.endswith("_e"):
            dataset = dataset[:-2]

        max_length = max_length_global
        QUERY2PROMPT = {
                "narrativeqa": "Question: {input} ",
                "qasper": "Question: {input}",
                "multifieldqa_en": "Question: {input}",
                "multifieldqa_zh": "Question: {input}",
                "hotpotqa": "Question: {input}",
                "2wikimqa": "Question: {input}",
                "musique": "Question: {input}",
                "dureader": "Question: {input}",
                "gov_report": "Question: You are given a report by a government agency. Write a one-page summary of the report.",
                "qmsum": "Question: {input}",
                "multi_news": "Question: You are given several news passages. Write a one-page summary of all news.",
                "vcsum": "下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。",
                "trec": "Question: {input}",
                "triviaqa": "Question: {input}",
                "samsum": "Question: Summarize the dialogue into a few short sentences.",
                "lsht": "请判断给定新闻的类别",
                "passage_count": " Question 1: There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total? \n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.",
                "passage_retrieval_en": "Question 1: Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from. The following is an abstract. {input} Please enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.",
                "passage_retrieval_zh": "下面是一个摘要：{input}请输入摘要所属段落的编号。答案格式必须是\"段落1\"，\"段落2\"等格式",
                "lcc": "Question: Please complete the code given below.",
                "repobench-p": "Question: Please complete the code given below."
            }
        Query = "Next I will give you one or more questions that will guide you to extract key information from the following document to answer more of the corresponding questions.\n"
        
        input_copy = input
        if "Question:\n" in input:
            input = input.split("Question:\n")[-1]
        if "Question: " in input:
            input = input.split("Question: ")[-1]
        if "\nAnswer:" in input:
            input = input.split("\nAnswer:")[0]
        if "\nSummary:" in input:
            input = "Summary the page "
        if "\nType:" in input: #"类别:"
            input = input.split("\nType:")[0]
        if "\n类别：" in input: #"类别:"
            input = input.split("\n类别：")[0]
        query = QUERY2PROMPT[dataset].format(input=input)
        Query = f"{Query}{query}"
        Query = Query + " The question given is the end. \n\n"
       
        encoded_query = tokenizer(Query,add_special_tokens=False)['input_ids']
       
        max_length = max_length - len(encoded_query)
       
        prompt_template = prompt_templates[dataset]
        prompt = prompt_template.format(input=input_copy, context=context)

        if truncate_from_middle:
            tokenized_prompt = tokenizer.encode(prompt)
            if len(tokenized_prompt) > max_length:
                half = int(max_length / 2)
                prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        else:
            tokenized_prompt = tokenizer.encode(prompt)
            prompt = tokenizer.decode(tokenized_prompt[-max_length:], skip_special_tokens=True)

        # in fewshot learning and code completion we do not need chat template
        if not any(x in DATASET2CATEGORY[dataset] for x in ["Few-Shot Learning", "Code Completion"]):
            prompt = apply_chat_template(
                chat_template,
                messages=[{'role': 'user', 'content': prompt}],
                tokenizer=tokenizer,
                add_generation_prompt=True,
            ).raw
            prompt = f"[INST] {Query}{prompt.split('[INST]')[-1]}"
            start = [4]
        else:
            prompt = Query + prompt
            start = [1]
        encoded = tokenizer(prompt)
        encoded["query_length"] = [len(encoded_query)]
        encoded["query"] = encoded_query
        encoded["query_start"] = start
        pre = input

        for k, v in encoded.items():
            outputs[k].append(v)
        outputs["dataset"].append(dataset)
        outputs["index"].append(index)

    return outputs


@torch.no_grad()
def main():
    parser = HfArgumentParser([Args])
    args = parser.parse_args_into_dataclasses()[0]

    accelerator = Accelerator(cpu=args.cpu)
    model, tokenizer = get_model_and_tokenizer(args, device=accelerator.device)

    # stop generation for QA tasks when \n appears
    if hasattr(model, "generation_config"):
        eos_token_id = model.generation_config.eos_token_id
    else:
        eos_token_id = tokenizer.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id.append(tokenizer.encode("\n", add_special_tokens=False)[-1])
    print(args.max_length)
    with accelerator.main_process_first():
        process_fn = partial(process_longbench,
            tokenizer=tokenizer,
            chat_template=args.chat_template,
            max_length_global=args.max_length,
            prompt_templates=DATASET2PROMPT,
            truncate_from_middle=args.truncate_from_middle,
        )

        raw_dataset = datasets.load_dataset("json", data_files=args.eval_data, cache_dir=args.dataset_cache_dir, split="train")
        dataset = raw_dataset.map(process_fn, batched=True, num_proc=1, with_indices=True, remove_columns=raw_dataset.column_names)
    groupby_dataset = dataset.to_pandas().groupby("dataset")

    metrics = {}
    if args.dataset_names is None:
        dataset_names = [key for key, _ in groupby_dataset]
    else:
        dataset_names = args.dataset_names

    result_dir = os.path.join(args.output_dir, args.result_dir)
    for i, dataset_name in enumerate(dataset_names):
        # if i<12:
            # continue
        if accelerator.process_index == 0:
            logger.info(f"Evaluating {dataset_name} ({i + 1} / {len(dataset_names)})...")

        result_path = os.path.join(result_dir, f"{dataset_name}.json")

        if args.load_result and os.path.exists(result_path):
            if accelerator.process_index == 0:
                with open(result_path, encoding="utf-8") as f:
                    score = json.loads(f.readline())
                logger.info(f"{dataset_name}: {score}")
                metrics[dataset_name] = score

        else:
            dataset = datasets.Dataset.from_pandas(groupby_dataset.get_group(dataset_name), preserve_index=False)

            data_collator = DefaultDataCollator(tokenizer=tokenizer)
            dataloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                collate_fn=data_collator,
                # only pin memory when no gpu
                pin_memory=not args.cpu,
            )

            if not args.enable_tp:
                # NOTE: prepare model only once
                if len(accelerator._models) == 0:
                    model, dataloader = accelerator.prepare(model, dataloader)
                    model = accelerator.unwrap_model(model)
                else:
                    dataloader = accelerator.prepare(dataloader)
            else:
                # NOTE: prepare dataloader so the data moves to GPU automatically
                dataloader = accelerator.prepare(dataloader)

            indices = []
            preds = []
            max_new_tokens = DATASET2MAXNEWTOKENS[dataset_name]

            for i, x in enumerate(tqdm(dataloader, desc="Generating")):
                x.pop("dataset")
                index = x.pop("index")[0]
                input_length = x["input_ids"].shape[1]
                # continue

                # NOTE: important to reset memory for every batch
                if hasattr(model, "memory"):
                    model.memory.reset()

                # NOTE: very important to include \n as an eos token for QA and trec, otherwise the F1 score is devastating
                if dataset_name in ["2wikimqa", "hotpotqa", "musique", "multifieldqa_en", "qasper", "narrativeqa", "samsum"]:
                    output = model.generate(
                        **x,
                        max_new_tokens=max_new_tokens,
                        do_sample=args.do_sample,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        eos_token_id=eos_token_id,
                        begin_suppress_tokens=eos_token_id,
                        # FIXME: sometimes transformers cannot detect deepspeed zero3, dont know why
                        synced_gpus=accelerator.state.deepspeed_plugin is not None and accelerator.state.deepspeed_plugin.zero_stage == 3,
                    )
                else:
                    output = model.generate(
                        **x,
                        max_new_tokens=max_new_tokens,
                        do_sample=args.do_sample,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        # FIXME: sometimes transformers cannot detect deepspeed zero3, dont know why
                        synced_gpus=accelerator.state.deepspeed_plugin is not None and accelerator.state.deepspeed_plugin.zero_stage == 3,
                    )

                # 1, max_new_tokens
                output = output[:, input_length:]
                if accelerator.num_processes > 1:
                    # pad across device to the same length
                    output = accelerator.pad_across_processes(output.contiguous(), pad_index=tokenizer.pad_token_id, dim=1)
                    # num_device, max_new_tokens
                    output = accelerator.gather_for_metrics(output)
                    index = accelerator.gather_for_metrics(index)

                output = output.tolist()
                index = index.tolist()

                if accelerator.process_index == 0:
                    pred = tokenizer.batch_decode(output, skip_special_tokens=True)
                    preds.extend(pred)
                    if isinstance(index, list):
                        indices.extend(index)
                    else:
                        # single process
                        indices.append(index)

            if accelerator.process_index == 0:
                raw_dataset_subset = raw_dataset[indices]
                answers = raw_dataset_subset["answers"]
                lengths = raw_dataset_subset["length"]
                all_classes = raw_dataset_subset["all_classes"][0]
                score = scorer(dataset_name, preds, answers, all_classes)

                logger.info(f"{dataset_name}: {score}")
                metrics[dataset_name] = score

                with open(makedirs(result_path), "w", encoding="utf-8") as f:
                    f.write(json.dumps(score, ensure_ascii=False) + "\n")
                    for index, pred in zip(indices, preds):
                        sample = raw_dataset[index]
                        del sample["all_classes"]
                        del sample["context"]
                        del sample["language"]
                        del sample["_id"]
                        sample["pred"] = pred
                        f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    if accelerator.process_index == 0:
        # save config
        args.save(os.path.join(result_dir, "config.json"))

        # compute category score
        category_metrics = defaultdict(list)
        for dataset, metric in metrics.items():
            category = DATASET2CATEGORY[dataset]
            category_metrics[category].append(metric)
        for k, v in category_metrics.items():
            # when evaluating on longbench_e, each metric is a dict of float
            if isinstance(v[0], dict):
                category_metric = {}
                for kk in v[0].keys():
                    vv = [v[j][kk] for j in range(len(v))]
                    category_metric[kk] = round(sum(vv) / len(vv), 2)
                category_metrics[k] = category_metric
            else:
                category_metrics[k] = round(sum(v) / len(v), 2)

        # compute average score
        if isinstance(next(iter(metrics.values())), dict):
            avg = defaultdict(list)
            for k, v in metrics.items():
                for kk, vv in v.items():
                    avg[kk].append(vv)
            for k, v in avg.items():
                avg[k] = round(sum(v) / len(v), 2)
        else:
            avg = round(sum(metrics.values()) / len(metrics), 2)
        metrics["avg"] = avg

        file_logger = FileLogger(makedirs(os.path.join(args.output_dir, "metrics.log")))
        file_logger.log(metrics, Args=asdict(args), Category_Metrics=category_metrics)


if __name__ == "__main__":
    main()
