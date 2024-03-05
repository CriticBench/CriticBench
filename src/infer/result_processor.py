import multiprocess
from sklearn.metrics import f1_score
from collections import defaultdict

import evaluate_configuration
from utils.parser import extract_answer_str_by_answer_pattern, extract_answer_by_question_source
from utils.grader import EM, math_equal
from utils.code_eval.code_eval import compute_code_eval

class ResultProcessor:
    def result_check(self, task, dataset_with_prompt, result_list, enable_code_execution):
        result_by_id = {sample["id"]: sample for sample in result_list}
        for sample in dataset_with_prompt:
            q_id = sample["id"]
            question = sample["question"]
            if task in ["generation", "correction"]:
                golden_answers = sample["answer"]
            elif task in ["critique"]:
                golden_answer = sample["response_label"]
            dataset_name = sample["question_source"]
            if q_id in result_by_id:
                pred_str = result_by_id[q_id][f"{task}_result"]
                answer_pattern = sample["prompt_info_dict"]["answer_pattern"]
                answer_str = extract_answer_str_by_answer_pattern(pred_str=pred_str, answer_pattern=answer_pattern)
                clean_pred = extract_answer_by_question_source(pred_str=answer_str, question_source=dataset_name, task=task)
                if task in ["generation", "correction"]:
                    label = False
                    if dataset_name in ["HotpotQA", "AmbigNQ"]:
                        for g in golden_answers:
                            if EM(clean_pred, g):
                                label = True
                    elif dataset_name in ["AQuA", "Colored Object", "CSQA", "Date", "Penguins"]:  # 选择题
                        for g in golden_answers:
                            if math_equal(g.lower(), clean_pred.lower()):
                                label = True
                    elif dataset_name == "HumanEval":
                        if "def" not in pred_str:
                            pred_str = question + pred_str
                        if task == "correction":
                            result_by_id[q_id][f"{task}_result"] = ((pred_str).replace("[DONE]", "")
                                                                    .replace("[BEGIN]", "")
                                                                    .replace("```python", "")
                                                                    .replace("```", ""))
                        else:
                            result_by_id[q_id][f"{task}_result"] = pred_str
                        label = False
                    elif dataset_name == "MBPP":
                        result_by_id[q_id][f"{task}_result"] = pred_str.replace("[DONE]", "").replace("[BEGIN]", "")
                        label = False
                    elif dataset_name == "Repeat Copy":
                        result_by_id[q_id][f"{task}_result"] = pred_str.strip()
                        clean_pred = pred_str.strip()
                        for g in golden_answers:
                            if math_equal(clean_pred, g):
                                label = True
                    else:  # GSM8K, MATH, Object Counting, StrategyQA, TabMWP
                        for g in golden_answers:
                            if isinstance(g, str):
                                g = g.lower()
                            if isinstance(clean_pred, str):
                                clean_pred = clean_pred.lower()
                            if math_equal(clean_pred, g):
                                label = True
                    result_by_id[q_id][f"{task}_check"] = label
                elif task in ["critique"]:
                    result_by_id[q_id][f"{task}_check"] = [golden_answer, clean_pred]
            else:
                result_by_id[q_id] = {"id":q_id, f"{task}_result": "", f"{task}_check": False}
        if enable_code_execution and task in ["generation", "correction"]:
            result_by_id = self.check_code(task=task, dataset_with_prompt=dataset_with_prompt, result_by_id=result_by_id)
        return result_by_id

    def critic_model_result_check(self, model, task, dataset, result_list):
        result_by_id = {sample["id"]: sample for sample in result_list}
        for sample in dataset:
            q_id = sample["id"]
            golden_answer = sample["response_label"]
            if q_id in result_by_id:
                pred_str = result_by_id[q_id][f"{task}_result"]
                clean_pred = None
                if ("incorrect" in pred_str or "wrong" in pred_str or "incomplete" in pred_str or "not helpful" in pred_str or
                        "error" in pred_str):
                    clean_pred = False
                elif "correct" in pred_str or "accurate" in pred_str or "good" in pred_str or "concise" in pred_str:
                    clean_pred = True
                elif "autoj-13b" in model:
                    try:
                        if "Rating: [[" in pred_str:
                            pos = pred_str.rfind("Rating: [[")
                            pos2 = pred_str.find("]]", pos)
                            assert pos != -1 and pos2 != -1
                            pred_score = float(pred_str[pos + len("Rating: [["):pos2].strip())
                            if pred_score >= 6:
                                clean_pred = True
                            else:
                                clean_pred = False
                    except (ValueError, TypeError):
                        clean_pred = not golden_answer
                elif "UltraCM-13b" in model:
                    try:
                        if "/" in pred_str:
                            pred_str = pred_str.split("/")[0]
                        pred_score = int(pred_str)
                        if pred_score >= 6:
                            clean_pred = True
                        else:
                            clean_pred = False
                    except (ValueError, TypeError):
                        clean_pred = not golden_answer
                if clean_pred is None:
                    clean_pred = not golden_answer
                result_by_id[q_id][f"{task}_check"] = [golden_answer, clean_pred]
            else:
                result_by_id[q_id] = {"id":q_id, f"{task}_result": "", f"{task}_check": False}
        return result_by_id

    def check_code(self, task, dataset_with_prompt, result_by_id):
        predictions = []
        references = []
        for sample in dataset_with_prompt:
            q_id = sample["id"]
            golden_answer = sample["answer"]
            if sample["prompt_info_dict"]["answer_pattern"] in ["code", "json: code"]:
                final_code = result_by_id[q_id][f"{task}_result"]
                predictions.append([q_id, final_code])
                references.append(golden_answer[0])
        num_worker = multiprocess.cpu_count()
        results, detailed_result = compute_code_eval(predictions=predictions,
                          references=references,
                          num_workers=num_worker)
        for q_id in detailed_result:
            result_by_id[q_id][f"{task}_check"] = detailed_result[q_id][0][1]["passed"]

        return result_by_id

    def analyse_result(self, task, dataset_with_prompt, result_by_id):
        check_by_type = defaultdict(list)
        check_by_dataset = defaultdict(list)
        total_check = []
        score_by_type = defaultdict(float)
        score_by_dataset = defaultdict(float)
        total_score = 0.0
        for sample in dataset_with_prompt:
            q_id = sample["id"]
            dataset_name = sample["question_source"]
            question_type = sample["question_type"]
            if q_id in result_by_id:
                check = result_by_id[q_id][f"{task}_check"]
                check_by_type[question_type].append(check)
                check_by_dataset[dataset_name].append(check)
                total_check.append(check)
        if task in ["generation", "correction"]:
            for type in check_by_type:
                score_by_type[type] = 100 * sum(check_by_type[type]) / len(check_by_type[type])
            for dataset in check_by_dataset:
                score_by_dataset[dataset] = 100 * sum(check_by_dataset[dataset]) / len(check_by_dataset[dataset])
            total_score = 100 * sum(total_check) / len(total_check)
        elif task in ["critique"]:  # F1 score of wrong label
            for type in check_by_type:
                golden_labels = [not check[0] for check in check_by_type[type]]
                pred_labels = []
                for check in check_by_type[type]:
                    if check == "" or check[1] == "":
                        pred_labels.append(check[0])
                    else:
                        pred_labels.append(not check[1])
                score_by_type[type] = 100 * f1_score(y_true=golden_labels, y_pred=pred_labels)
            for dataset in check_by_dataset:
                golden_labels = [not check[0] for check in check_by_dataset[dataset]]
                pred_labels = []
                for check in check_by_dataset[dataset]:
                    if check == "" or check[1] == "":
                        pred_labels.append(check[0])
                    else:
                        pred_labels.append(not check[1])
                score_by_dataset[dataset] = 100 * f1_score(y_true=golden_labels, y_pred=pred_labels)
            golden_labels = [not check[0] for check in total_check]
            pred_labels = []
            for check in total_check:
                if check == "" or check[1] == "":
                    pred_labels.append(check[0])
                else:
                    pred_labels.append(not check[1])
            total_score = 100 * f1_score(y_true=golden_labels, y_pred=pred_labels)
        return total_score, score_by_type, score_by_dataset

