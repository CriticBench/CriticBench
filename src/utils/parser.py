"""
The code implementation mainly refers to https://github.com/microsoft/ToRA/tree/main
"""
import json
import re


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        if "sqrt" not in a:
            a = int(a)
        if "sqrt" not in b:
            b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string


def _fix_sqrt(string):
    _string = re.sub(r"\\sqrt(\w+)", r"\\sqrt{\1}", string)
    return _string


def strip_string(string):
    string = str(string).strip()
    string = string.replace("\n", "")

    string = string.rstrip(".")

    string = string.replace("\\!", "")
    string = string.replace("\\ ", "")

    string = string.replace("\\\\", "\\")
    string = string.replace("\\\\", "\\")

    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    _string = re.sub(r"\\text{.*?}$", "", string).strip()
    if _string != "" and _string != string:
        string = _string

    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    string = string.replace("\\$", "")
    string = string.replace("$", "")

    string = string.replace("\\text", "")
    string = string.replace("x\\in", "")

    string = string.replace("\\%", "")
    string = string.replace("\%", "")
    string = string.replace("%", "")

    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    string = string.replace("\\cdot", "")

    string = string.replace("infinity", "\\infty")
    if "\\infty" not in string:
        string = string.replace("inf", "\\infty")
    string = string.replace("+\\inity", "\\infty")

    string = string.replace("and", "")
    string = string.replace("\\mathbf", "")

    string = re.sub(r"\\mbox{.*?}", "", string)

    string.replace("'", "")
    string.replace("\"", "")

    if "j" in string and "i" not in string:
        string = string.replace("j", "i")

    string = re.sub(r"(\d+)\.0+([^\d])", r"\1\2", string)
    string = re.sub(r"(\d+)\.0+$", r"\1", string)

    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    string = _fix_sqrt(string)
    string = string.replace(" ", "")

    string = _fix_fracs(string)

    string = _fix_a_slash_b(string)

    return string


def extract_latex_answer(answer_string):
    if 'boxed' in answer_string:
        ans = answer_string.split('boxed')[-1]
        if len(ans) == 0:
            return ""
        elif (ans[0] == '{'):
            stack = 1
            a = ''
            for c in ans[1:]:
                if (c == '{'):
                    stack += 1
                    a += c
                elif (c == '}'):
                    stack -= 1
                    if (stack == 0): break
                    a += c
                else:
                    a += c
        else:
            a = ans.split('$')[0].strip()
        pred = a
    elif 'he answer is' in answer_string:
        pred = answer_string.split('he answer is')[-1].strip()
    else:
        pred_str = answer_string.replace("$", "").replace("€", "").replace(",", "").replace("£", "").replace(", ", "")
        pattern = '-?\d*\.?\d+'
        pred = re.findall(pattern, pred_str)
        if (len(pred) >= 1):
            pred = pred[-1]
        else:
            pred = ''
    pred = pred.split("\n")[0]
    if pred != "" and pred[0] == ":":
        pred = pred[1:]
    if pred != "" and pred[-1] == ".":
        pred = pred[:-1]
    if pred != "" and pred[-1] == "/":
        pred = pred[:-1]
    pred = strip_string(pred)
    return pred


def extract_answer_str_by_answer_pattern(pred_str, answer_pattern):
    pred_result = pred_str
    if "json" in answer_pattern:
        first_dict_list = re.findall("\{.*?}", pred_str, re.DOTALL)
        key = answer_pattern.replace("json:", "").strip()
        if len(first_dict_list) > 0 and key in first_dict_list[0]:
            pred_str = first_dict_list[0]
        try:
            result_dict = json.loads(pred_str)
            pred_result = str(result_dict[key])
        except (json.JSONDecodeError, KeyError, TypeError):
            key_list = re.findall(f"{key}.*?" + "}", pred_str, re.DOTALL)
            if len(key_list) > 0:
                pred_result = key_list[0]
            else:
                pred_result = pred_str
    elif "[[]]" == answer_pattern:
        p_list = re.findall('\[\[(.*?)]]', pred_str)
        if len(p_list) > 0:
            pred_result = p_list[-1]
    return pred_result


def extract_answer_by_question_source(pred_str, question_source, task):
    pred = ""
    if task in ["generation", "correction"]:
        if question_source in ["MATH"]:
            pred = extract_latex_answer(pred_str)
        elif question_source in ["MBPP", "HumanEval"]:
            pred = pred_str
        elif question_source in ["AmbigNQ"]:
            if "he answer is" in pred_str:
                pred_str = pred_str.split("he answer is:")[-1]
                pred = pred_str.strip(".")
            else:
                pred = pred_str.strip(".")
        elif question_source in ["StrategyQA"]:
            if "he answer is " in pred_str:
                pred_str = pred_str.split("he answer is ")[-1].lower()
                if "yes" in pred_str and "no" not in pred_str:
                    pred = "yes"
                elif "yes" not in pred_str and "no" in pred_str:
                    pred = "no"
            else:
                pred_str = pred_str.lower()
                if re.search(r'\byes\b', pred_str) and not re.search(r'\bno\b', pred_str):
                    pred = "yes"
                elif not re.search(r'\byes\b', pred_str) and re.search(r'\bno\b', pred_str):
                    pred = "no"
        elif question_source in ["CSQA", "Date", "Penguins", "Colored Object"]:
            if "he answer is " in pred_str:
                pred = ""
                pred_str = pred_str.split("he answer is ")[-1]
                option_list = re.findall("\((.)\)", pred_str)
                if len(option_list) > 0:
                    pred = option_list[-1]
            else:
                option_list = re.findall("\((.)\)", pred_str)
                if len(option_list) > 0:
                    pred = option_list[-1]
                else:
                    pattern = r'\b(?:option: |answer: )?\(?([A-Za-z])\)?(?=[\s\.,:;]|$)'
                    option_list = re.findall(pattern, pred_str)
                    if len(option_list) > 0:
                        pred = option_list[-1]
        elif question_source in ["TabMWP"]:
            if "he answer is " in pred_str:
                pred_str = pred_str.split("he answer is ")[-1]
            option_list = re.findall("\((.)\)", pred_str)
            if len(option_list) > 0:
                pred = option_list[-1]
            else:
                pred_str = (pred_str.replace("$", "").replace("€", "").replace(",", "").replace("£", "")
                            .replace(", ", "").replace(" /", "/").replace("/ ", "/"))
                if len(re.findall("(\d+/\d+)", pred_str)) > 0:
                    pred = re.findall("(\d+/\d+)", pred_str)[-1]
                    pred_str = pred_str.split(pred)[-1]
                p_list = re.findall('-?\d*\.?\d+', pred_str)
                if len(p_list) > 0:
                    pred = p_list[-1]
                else:
                    pred = pred_str.strip(".")
                if len(p_list) > 0:
                    pred = p_list[-1]
            if pred == "":  # string answer
                pred = pred_str
        elif question_source in ["HotpotQA"]:
            if "he answer is" in pred_str:
                pred_str = pred_str.split("he answer is")[-1]
                pred = pred_str.strip(".")
            else:
                pred = pred_str.strip(".")
        elif question_source in ["AQuA"]:
            patterns = [
                'he answer is (.)\).',
                r'\b(.)\)',
                r'\b[A-G]\b',
                r'\b(?:option: |answer: )?\(?([A-G])\)?(?=[\s\.,:;]|$)'
            ]
            for pattern in patterns:
                p_list = re.findall(pattern, pred_str)
                if len(p_list) > 0:
                    pred = p_list[-1]
                    break
        elif question_source in ["GSM8K", "Object Counting"]:
            if "he answer is " in pred_str:
                pred_str = pred_str.split("he answer is ")[-1]
                pred_str = pred_str.replace("$", "").replace("€", "").replace(",", "").replace("£", "").replace(", ",
                                                                                                                "")
            else:
                pred_str = pred_str.replace("$", "").replace("€", "").replace(",", "").replace("£", "").replace(", ",
                                                                                                                "")
            p_list = re.findall('-?\d*\.?\d+', pred_str)
            if len(p_list) > 0:
                pred = p_list[-1]
        elif question_source in ["Repeat Copy"]:
            pred = pred_str
    elif task == "critique":
        pred_str = pred_str.lower()
        if "[[correct]]" in pred_str and "[[wrong]]" not in pred_str:
            pred = True
        elif "[[correct]]" not in pred_str and "[[wrong]]" in pred_str:
            pred = False
        elif "wrong" in pred_str or "incorrect" in pred_str or "not correct" in pred_str:
            pred = False
        elif "correct" in pred_str:
            pred = True
    return pred
