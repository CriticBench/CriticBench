import json
import os
import evaluate_configuration
from datasets import load_dataset

class DataProcessor:
    def __init__(self, prompt_type_by_task, data_cache_dir="./data", prompt_dir="./prompt"):
        self.data_cache_dir = data_cache_dir
        self.prompt_dir = prompt_dir
        self.dataset = self.load_data()
        gen_prompt_type = prompt_type_by_task["G"]
        crit_prompt_type = prompt_type_by_task["Q"]
        corr_prompt_type = prompt_type_by_task["C"]
        self.prompt_file_by_task= {
            "generation":evaluate_configuration.prompt_type_to_file_name["generation"][gen_prompt_type],
            "critique":evaluate_configuration.prompt_type_to_file_name["critique"][crit_prompt_type],
            "correction":evaluate_configuration.prompt_type_to_file_name["correction"][corr_prompt_type]
        }
        self.prompt_info_by_task = {
            "generation": None,
            "critique": None,
            "correction": None
        }
        self.prepare_all_prompt()


    def load_data(self):
        if not os.path.exists(self.data_cache_dir):
            os.makedirs(self.data_cache_dir, exist_ok=True)
        dataset = load_dataset("llm-agents/CriticBench", cache_dir=self.data_cache_dir)
        return [example for example in dataset["test"]]

    def prepare_all_prompt(self):
        def load_prompt_of_task(task):
            all_prompt = {}
            dataset_list = evaluate_configuration.dataset_in_criticbench
            for dataset in dataset_list:
                prompt_file = os.path.join(self.prompt_dir, task, dataset, self.prompt_file_by_task[task])
                all_prompt[dataset] = self.load_single_prompt(prompt_file)
            return all_prompt
        for task in self.prompt_info_by_task:
            self.prompt_info_by_task[task] = load_prompt_of_task(task)


    def load_single_prompt(self, prompt_file):
        instruction = ""
        examples = ""
        question = ""
        answer_pattern = ""
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt_string = f.read()
            prompt_parts = prompt_string.split("\n# ########## #")
            for part in prompt_parts:
                if "#instruction#" in part:
                    instruction = part.replace("#instruction#\n", "").replace("#instruction#", "")
                elif "#Example#" in part:
                    examples = part.replace("#Example#\n", "").replace("#Example#", "")
                elif "#Question#" in part:
                    question = part.replace("#Question#\n", "").replace("#Question#", "")
                elif "#Answer Format#" in part:
                    answer_pattern = part.replace("#Answer Format#", "").strip()
        if len(examples.strip()) == 0:
            examples = ""
        return {"instruction":instruction, "examples":examples, "question_template": question, "answer_pattern": answer_pattern}

    def prepare_data_with_prompt_info(self, task):
        dataset_with_prompt = []
        for line in self.dataset:
            question_source = line["question_source"]
            prompt_info_dict = self.prompt_info_by_task[task][question_source]
            line["prompt_info_dict"] = prompt_info_dict
            dataset_with_prompt.append(line)
        return dataset_with_prompt



