import os
import re
import json
import time
from vllm import SamplingParams
from transformers import AutoTokenizer, AutoConfig
import openai
import tiktoken

import evaluate_configuration
from model_template import UltraCM, AutoJ




def prepare_question_with_template(sample, question_template, task):
    question = sample["question"]
    if task == "generation":
        blank_list = re.findall("\{question}", question_template)
        context_for_blank = [question]
    elif task == "critique":
        response = sample["response"]
        dataset = sample["question_source"]
        if dataset == "HumanEval":
            blank_list = re.findall("\{solution}", question_template)
            context_for_blank = [response]
        else:
            blank_list = re.findall("\{question}|\{solution}", question_template)
            context_for_blank = [question, response]
    elif task == "correction":
        response = sample["response"]
        critique = sample["critique"]
        dataset = sample["question_source"]
        if dataset == "HumanEval":
            blank_list = re.findall("\{solution}|\{critique}", question_template)
            context_for_blank = [response, critique]
        else:
            blank_list = re.findall("\{question}|\{solution}|\{critique}", question_template)
            context_for_blank = [question, response, critique]
        pass
    assert len(blank_list) == len(context_for_blank)
    for b, c in zip(blank_list, context_for_blank):
        question_template = question_template.replace(b, c)
    return question_template

def set_prompt(model_path, task, max_gen_len,  dataset_with_prompt, model_from_api=False):
    if model_from_api:
        tokenizer = tiktoken.encoding_for_model(model_path)
        limit_length = evaluate_configuration.model_from_api_limit_length[model_path]
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        limit_length = config.max_position_embeddings
    for sample in dataset_with_prompt:
        question = prepare_question_with_template(sample=sample,
                                                  question_template=sample["prompt_info_dict"]["question_template"],
                                                  task=task)
        instruction = sample["prompt_info_dict"]["instruction"]
        question_source = sample["question_source"]
        split_sep = evaluate_configuration.few_shot_split_sep_by_task[task][question_source]
        example_list = (sample["prompt_info_dict"]["examples"].split(split_sep))
        final_prompt = instruction + question
        model_template_len = 0
        for i in range(len(example_list)):
            examples = split_sep.join(example_list[:i+1])
            final_prompt = instruction + examples + question
            input_length = len(tokenizer.encode(final_prompt))
            if input_length > limit_length - max_gen_len - model_template_len:
                if split_sep == "\n---":
                    question = "\n---\n" + question
                final_prompt = instruction + split_sep.join(example_list[:i]) + question
                break
        final_prompt = final_prompt.strip()
        sample["final_prompt"] = final_prompt.strip()
    return dataset_with_prompt


def infer_openai(model, api_key, dataset_with_prompt, out_dir, task, prompt_type):
    max_gen_len = 512
    model_name = model.split("/")[-1]
    openai.api_key = api_key
    save_dir = os.path.join(out_dir, model_name, task)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    dataset_with_prompt = set_prompt(model_path=model, task=task, max_gen_len=max_gen_len,
                                     dataset_with_prompt=dataset_with_prompt, model_from_api=True)
    print(f"----------Start {task}----------")
    print("Number of samples: ", len(dataset_with_prompt))
    save_path = os.path.join(save_dir,
                             f"{prompt_type}_result_{time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))}.jsonl")
    result_list = []
    with open(save_path, "w", encoding="utf-8") as f:
        for sample in dataset_with_prompt:
            final_prompt = sample["final_prompt"]
            retry_delay_seconds = 5
            for i in range(5):
                try:
                    response = openai.ChatCompletion.create(
                        model=model,
                        messages=[{"role": "user", "content": final_prompt}],
                        max_tokens=max_gen_len,
                        temperature=0,
                        top_p=0.95,
                    )
                    if "choices" in response.keys():
                        result = {"id":sample["id"], "final_prompt":sample["final_prompt"],
                                f'{task}_result': response["choices"][0]["message"]["content"]}
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                        break
                    else:
                        print(response)
                except (openai.APIError,openai.error.RateLimitError) as e:
                    print(e)
                    time.sleep(retry_delay_seconds)
                    retry_delay_seconds *= 2
        result_list.append(result)
    return result_list

def infer_hf(model, llm, dataset_with_prompt, out_dir, task, prompt_type):
    max_gen_len = 512
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=max_gen_len, n=1,
                                     stop=["\n\nQuestion:",
                                           "\n\nTable:",
                                           "\n\n---",
                                           "\n\nYou are an expert Python programmer",
                                           "\nAnalysis and verdict:"]
                                     )
    model_name = model.split("/")[-1]
    save_dir = os.path.join(out_dir, model_name, task)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    dataset_with_prompt = set_prompt(model_path=model, task=task, max_gen_len=max_gen_len,
                                     dataset_with_prompt=dataset_with_prompt, model_from_api=False)
    print("Number of samples: ", len(dataset_with_prompt))
    inputs = [sample["final_prompt"] for sample in dataset_with_prompt]
    outputs = llm.generate(inputs, sampling_params)
    outputs = sorted(outputs, key=lambda x: int(x.request_id))
    outputs = [output.outputs[0].text for output in outputs]

    result_list = [{"id":sample["id"], "final_prompt":sample["final_prompt"],
                    f'{task}_result': output} for sample, output in zip(dataset_with_prompt, outputs)]

    save_path = os.path.join(save_dir, f"{prompt_type}_result_{time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))}.jsonl")

    with open(save_path, "w", encoding="utf-8") as f:
        for sample in result_list:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    return result_list


def set_critic_model_prompt(model, dataset):
    for sample in dataset:
        question = sample["question"]
        response = sample["response"]
        if "UltraCM-13b" in model:
            final_prompt = UltraCM.get_prompt(question=question, response=response)
        elif "autoj-13b" in model:
            final_prompt = AutoJ.get_prompt(question=question, response=response)
        sample["final_prompt"] = final_prompt.strip()
    return dataset


def infer_hf_critic_model(model, llm, dataset, out_dir, task):
    max_gen_len = 512
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=max_gen_len, n=1)
    model_name = model.split("/")[-1]
    print("Number of samples: ", len(dataset))

    save_dir = os.path.join(out_dir, model_name, task)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    dataset_with_prompt = set_critic_model_prompt(model, dataset)

    inputs = [sample["final_prompt"] for sample in dataset_with_prompt]
    outputs = llm.generate(inputs, sampling_params)
    outputs = sorted(outputs, key=lambda x: int(x.request_id))
    outputs = [output.outputs[0].text for output in outputs]

    result_list = [{"id":sample["id"], "final_prompt":sample["final_prompt"],
                    f'{task}_result': output} for sample, output in zip(dataset_with_prompt, outputs)]

    save_path = os.path.join(save_dir, f"result_{time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))}.jsonl")

    with open(save_path, "w", encoding="utf-8") as f:
        for sample in result_list:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    return result_list