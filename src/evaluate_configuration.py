openai_api_key = ""
hf_token = ""

dataset_in_criticbench = [
    "GSM8K", "MATH", "AQuA", "CSQA", "AmbigNQ", "StrategyQA",
    "HotpotQA", "Penguins", "Colored Object", "Date", "MBPP",
    "Object Counting", "Repeat Copy", "TabMWP", "HumanEval"
]

model_from_api_limit_length = {
    "gpt-3.5-turbo-0613": 4096,
    "gpt-4": 8192
}

full_task_name_dict = {
    "G": "generation",
    "Q": "critique",
    "C": "correction"
}

few_shot_split_sep_by_task = {
    "generation": {
        "GSM8K": "\nQuestion:",
        "MATH": "\nQuestion:",
        "AQuA": "\nQuestion:",
        "CSQA": "\nQuestion:",
        "AmbigNQ": "\nQuestion:",
        "StrategyQA": "\nQuestion:",
        "HotpotQA": "\nQuestion:",
        "Penguins": "\nQuestion:",
        "Colored Object": "\nQuestion:",
        "Date": "\nQuestion:",
        "MBPP": "\nYou are an expert Python programmer",
        "Object Counting": "\nQuestion:",
        "Repeat Copy": "\nQuestion:",
        "TabMWP": "\nTable:",
        "HumanEval": "\n---"
    },
    "critique": {
        "GSM8K": "\n---",
        "MATH": "\n---",
        "AQuA": "\n---",
        "CSQA": "\n---",
        "AmbigNQ": "\n---",
        "StrategyQA": "\n---",
        "HotpotQA": "\n---",
        "Penguins": "\n---",
        "Colored Object": "\n---",
        "Date": "\n---",
        "MBPP": "\n---",
        "Object Counting": "\n---",
        "Repeat Copy": "\n---",
        "TabMWP": "\n---",
        "HumanEval": "\n---"
    },
    "correction": {
        "GSM8K": "\n---",
        "MATH": "\n---",
        "AQuA": "\n---",
        "CSQA": "\n---",
        "AmbigNQ": "\n---",
        "StrategyQA": "\n---",
        "HotpotQA": "\n---",
        "Penguins": "\n---",
        "Colored Object": "\n---",
        "Date": "\n---",
        "MBPP": "\n---",
        "Object Counting": "\n---",
        "Repeat Copy": "\n---",
        "TabMWP": "\n---",
        "HumanEval": "\n---"
    }
}

dataset_answer_format_dict = {
    "AmbigNQ": "So the answer is:",
    "AQuA": "The answer is ).",
    "CSQA": "So the answer is ().",
    "GSM8K": "The answer is",
    "HotpotQA": "The answer is string",
    "MATH": "The answer is latex",
    "StrategyQA": "So the answer is yes/no.",
    "TabMWP": "The answer is mixed format",
    "Date": "So the answer is ().",
    "Object Counting": "The answer is",
    "Penguins": "So the answer is ().",
    "Repeat Copy": "full",
    "Colored Object": "So the answer is ().",
    "MBPP": "code",
    "HumanEval": "code"
}

prompt_type_to_file_name = {
    "generation": {
        "fs": "few_shot.txt",
    },
    "critique": {
        "fs": "few_shot.txt",
        "zs": "zero_shot_chain_of_thought.txt",
        "zs-crit-cot": "zero_shot_chain_of_thought.txt",
        "zs-crit-ao-1": "zero_shot_answer_only_1.txt",
        "zs-crit-ao-2": "zero_shot_answer_only_2.txt",
        "zs-crit-ao-3": "zero_shot_answer_only_3.txt",
    },
    "correction": {
        "fs": "few_shot.txt",
        "zs": "zero_shot_chain_of_thought.txt",
        "zs-cot": "zero_shot_with_chain_of_thought_critique.txt",
        "zs-ao": "zero_shot_with_answer_only_critique.txt"
    }
}
