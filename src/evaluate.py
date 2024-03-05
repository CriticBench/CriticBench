import argparse
import json
import os
import sys
import warnings
from huggingface_hub import login
from vllm import LLM

from utils.data_processor import DataProcessor
from utils.utils import load_jsonl
import evaluate_configuration
from infer.infer import infer_hf, infer_openai, infer_hf_critic_model
from infer.result_processor import ResultProcessor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=str, default="GQC", choices=["GQC", "QC", "G", "Q", "C"],
                        help="Specify the evaluation task. Options: G (generation), Q (critique), C (correction),"
                             "The correction task must be executed after the critique, "
                             "or a critique result file must be specified.")
    parser.add_argument("--hf_model", type=str, default=None,
                        help="Path or identifier for a HuggingFace model.")
    parser.add_argument("--hf_critic_model", type=str, default=None,
                        help="Identifier for a critique model from Hugging Face, "
                             "supporting GAIR/autoj-13b and openbmb/UltraCM-13b models.")
    parser.add_argument("--openai_model", type=str, default=None,
                        help="Model from openai")
    parser.add_argument("--enable_code_execution", action="store_true",
                        help="Enable code execution for generation and correction.")

    parser.add_argument("--available_gpus", type=str, default="0",
                        help="Specifies which GPUs to use, by ID, separated by commas (e.g., '0,1').")
    # parser.add_argument("--enable_vllm", action="store_true", default=True)
    parser.add_argument("--prompt_type", type=str, default="fs",
                        choices=["fs", "zs-crit-cot", "zs-crit-ao-1", "zs-crit-ao-2", "zs-crit-ao-3"])

    parser.add_argument("--existed_gen_file", type=str, default=None,
                        help="Path to an existing file with generation results to be used or evaluated.")
    parser.add_argument("--existed_crit_file", type=str, default=None,
                        help="Path to an existing file with critique results to be used or evaluated.")
    parser.add_argument("--existed_corr_file", type=str, default=None,
                        help="Path to an existing file with correction results to be used or evaluated.")

    parser.add_argument("--prompt_dir", type=str, default="./prompt")
    parser.add_argument("--data_cache_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./output")
    args = parser.parse_args()
    return args

def set_environment(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.available_gpus
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    evaluate_configuration.openai_api_key = os.environ.get("OPENAI_API_KEY")
    evaluate_configuration.hf_token = os.environ.get("HUGGINGFACE_TOKEN")
    if args.enable_code_execution:
        os.environ["HF_ALLOW_CODE_EVAL"] = "1"


def set_existed_file(args):
    existed_file_by_task = {
        "G": args.existed_gen_file,
        "Q": args.existed_crit_file,
        "C": args.existed_corr_file,
    }
    has_existed_file = not all(value is None for value in existed_file_by_task.values())
    return existed_file_by_task, has_existed_file


def set_prompt_type_by_task(prompt_type):
    prompt_type_by_task = {t: "" for t in ["G", "Q", "C"]}
    if prompt_type == "fs":
        prompt_type_by_task = {t: "fs" for t in ["G", "Q", "C"]}
    elif prompt_type == "zs-crit-cot":
        prompt_type_by_task = {
            "G": "fs",
            "Q": prompt_type,
            "C": "zs-cot"
        }
    elif prompt_type in ["zs-crit-ao-1", "zs-crit-ao-2", "zs-crit-ao-3"]:
        prompt_type_by_task = {
            "G": "fs",
            "Q": prompt_type,
            "C": "zs-ao"
        }
    return prompt_type_by_task


def load_exist_result(existed_file):
    if os.path.exists(existed_file):
        print(f"evaluate result in {existed_file}")
        result_list = list(load_jsonl(existed_file))
        return result_list
    else:
        sys.exit(f"file:{existed_file} not existed")

def init_hf_model_with_vllm(model, available_gpus, llm):
    if llm == None:
        available_gpus = available_gpus.split(',')
        llm = LLM(model=model, tensor_parallel_size=len(available_gpus), trust_remote_code=True)
    return llm


def evaluate_model(args, existed_file_by_task, prompt_type_by_task, data_processor, result_processor):
    llm = None
    if args.hf_model is not None and evaluate_configuration.hf_token is not None:
        login(token=evaluate_configuration.hf_token)
    result_dict = {}
    for task in ["G", "Q", "C"]:
        if task in args.tasks:
            print(f"----------Start {evaluate_configuration.full_task_name_dict[task]}----------")
            dataset_with_prompt = data_processor.prepare_data_with_prompt_info(
                evaluate_configuration.full_task_name_dict[task])
            existed_file = existed_file_by_task[task]
            prompt_type = prompt_type_by_task[task]
            if existed_file is None:
                if args.hf_model is not None:
                    llm = init_hf_model_with_vllm(model=args.hf_model, available_gpus=args.available_gpus, llm=llm)
                    result_list = infer_hf(model=args.hf_model,
                                           llm=llm,
                                           dataset_with_prompt=dataset_with_prompt,
                                           out_dir=args.output_dir,
                                           task=evaluate_configuration.full_task_name_dict[task],
                                           prompt_type=prompt_type)
                elif args.openai_model is not None:
                    result_list = infer_openai(model=args.openai_model,
                                               api_key=evaluate_configuration.openai_api_key,
                                               dataset_with_prompt=dataset_with_prompt,
                                               out_dir=args.output_dir,
                                               task=evaluate_configuration.full_task_name_dict[task],
                                               prompt_type=prompt_type)
            else:
                if prompt_type not in existed_file:
                    warnings.warn("Mismatched prompt types may lead to issues with the extract answer function.",
                                  UserWarning)
                result_list = load_exist_result(existed_file=existed_file)
            result_by_id = result_processor.result_check(task=evaluate_configuration.full_task_name_dict[task],
                                                         dataset_with_prompt=dataset_with_prompt,
                                                         result_list=result_list,
                                                         enable_code_execution=args.enable_code_execution)
            total_score, score_by_type, score_by_dataset = (
                result_processor.analyse_result(task=evaluate_configuration.full_task_name_dict[task],
                                                dataset_with_prompt=dataset_with_prompt,
                                                result_by_id=result_by_id))
            result_dict[evaluate_configuration.full_task_name_dict[task]] = {
                f"{evaluate_configuration.full_task_name_dict[task]} score": total_score,
                f"{evaluate_configuration.full_task_name_dict[task]} score by type": score_by_type,
                f"{evaluate_configuration.full_task_name_dict[task]} score by dataset": score_by_dataset
            }
            if task == "Q":
                for sample in dataset_with_prompt:
                    id = sample["id"]
                    sample["critique"] = result_by_id[id]["critique_result"]
    print("----------Evaluation result----------")
    print(json.dumps(result_dict, indent=2))


def evaluate_critic_model(args, data_processor, result_processor):
    llm = None
    if "UltraCM-13b" or "autoj-13b" in args.hf_critic_model:
        if args.existed_crit_file is None:
            llm = init_hf_model_with_vllm(model=args.hf_critic_model, available_gpus=args.available_gpus, llm=llm)
            result_list = infer_hf_critic_model(model=args.hf_critic_model,
                                                llm=llm,
                                                dataset=data_processor.dataset,
                                                out_dir=args.output_dir,
                                                task="critique")
        else:
            result_list = load_exist_result(existed_file=args.existed_crit_file)
    result_by_id = result_processor.critic_model_result_check(model=args.hf_critic_model,
                                                              task="critique",
                                                              dataset=data_processor.dataset,
                                                              result_list=result_list)
    total_score, score_by_type = (
        result_processor.analyse_result(task="critique",
                                        dataset_with_prompt=data_processor.dataset,
                                        result_by_id=result_by_id))
    print(f"critique score: {total_score}")
    print(f"critique score by type: {json.dumps(score_by_type, indent=2)}")


if __name__ == '__main__':
    args = parse_args()
    set_environment(args)
    existed_file_by_task, has_existed_file = set_existed_file(args)
    prompt_type_by_task = set_prompt_type_by_task(prompt_type=args.prompt_type)
    data_processor = DataProcessor(data_cache_dir=args.data_cache_dir,
                                   prompt_dir=args.prompt_dir,
                                   prompt_type_by_task=prompt_type_by_task)
    result_processor = ResultProcessor()
    if args.hf_critic_model is not None:
        evaluate_critic_model(args=args,
                              data_processor=data_processor,
                              result_processor=result_processor)
    elif args.hf_model is not None or args.openai_model is not None or has_existed_file:
        evaluate_model(args=args,
                       existed_file_by_task=existed_file_by_task,
                       prompt_type_by_task=prompt_type_by_task,
                       data_processor=data_processor,
                       result_processor=result_processor)
