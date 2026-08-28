import glob
import json
import os
import re
import time
import pandas as pd
from prompt_lib import MMLU_QUESTION, COMPLEX_COT_EXAMPLES, TEMPERATURE, MAX_TOKENS
import openai
import backoff
try:
    from openai.error import (
        RateLimitError,
        APIError,
        ServiceUnavailableError,
        APIConnectionError,
        Timeout,
    )
    OPENAI_V1 = False
except ImportError:
    from openai import APIConnectionError, APIError, APITimeoutError, RateLimitError

    ServiceUnavailableError = APIConnectionError
    Timeout = APITimeoutError
    OPENAI_V1 = True

openai.api_key = "EMPTY"
openai.api_base = "http://localhost:8001/v1"


class OutOfQuotaException(Exception):

    def __init__(self, key, cause=None):
        super().__init__(f"No quota for key: {key}")
        self.key = key
        self.cause = cause

    def __str__(self):
        if self.cause:
            return f"{super().__str__()}. Caused by {self.cause}"
        else:
            return super().__str__()

class AccessTerminatedException(Exception):

    def __init__(self, key, cause=None):
        super().__init__(f"Access terminated key: {key}")
        self.key = key
        self.cause = cause

    def __str__(self):
        if self.cause:
            return f"{super().__str__()}. Caused by {self.cause}"
        else:
            return super().__str__()

def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string

def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string

def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) == 0:
                continue
            if substr[0] == "{":
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

def _remove_right_units(string):

    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) >= 2
        return splits[0]
    else:
        return string

def _strip_string(string):

    string = string.replace("\n", "")


    string = string.replace("\\!", "")


    string = string.replace("\\\\", "\\")


    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")


    string = string.replace("\\left", "")
    string = string.replace("\\right", "")


    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")


    string = string.replace("\\$", "")


    string = _remove_right_units(string)


    string = string.replace("\\%", "")
    string = string.replace("\%", "")


    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

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


    if string == "0.5":
        string = "\\frac{1}{2}"


    string = _fix_a_slash_b(string)

    return string

def parse_question_answer(df, ix):
    question = df.iloc[ix, 0]
    a = df.iloc[ix, 1]
    b = df.iloc[ix, 2]
    c = df.iloc[ix, 3]
    d = df.iloc[ix, 4]

    question = MMLU_QUESTION.format(question, a, b, c, d)

    answer = df.iloc[ix, 5]

    return question, answer

def get_mmlu_qa_pairs(csv_name):
    df = pd.read_csv(csv_name, header=None)
    ix = len(df)
    return [parse_question_answer(df, idx) for idx in range(ix)]

def get_mmlu_qa_pairs_from_dir(data_dir, limit=200):
    qa_pairs = []
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    for csv_name in csv_files:
        for que, ans in get_mmlu_qa_pairs(csv_name):
            qa_pairs.append((que, ans))
            if len(qa_pairs) >= limit:
                return qa_pairs
    return qa_pairs

def get_math_qa_pairs(sub_dir, min_file, max_file):
    def find_math_answer(s):
        assert('boxed' in s)

        ans = s.split('boxed')[-1]
        if(ans[0] == '{'):
            stack = 1
            a = ''
            for c in ans[1:]:
                if(c == '{'):
                    stack += 1
                    a += c
                elif(c == '}'):
                    stack -= 1
                    if(stack == 0): break
                    a += c
                else:
                    a += c
        else:
            a = ans.split('$')[0].strip()
        a=_strip_string(a)
        return a

    def parse_single_qa_math(subdir, file):
        with open(os.path.join(subdir, file), 'r') as fp:
            try:
                problem_data = json.load(fp)
            except Exception as e:
                print(f"Error loading JSON from {file}", e)
                raise e
            prob_content = problem_data["problem"]
            question = COMPLEX_COT_EXAMPLES + "\n\nPlease solve the problem below.\nProblem: " + prob_content + "\nAnswer:"
            prob_level = problem_data["level"]
            prob_type = problem_data["type"]
            try:
                prob_level = int(prob_level.split("Level ")[1])
            except:
                prob_level = None


            answer = find_math_answer(problem_data['solution'])

            return question, prob_level, prob_type, answer

    ret_list = []
    for subdir, dirs, files in os.walk(sub_dir):
        for file in files:
            file_num = int(os.path.splitext(file)[0])
            if min_file <= file_num <= max_file:
                question, prob_level, prob_type, answer = parse_single_qa_math(subdir, file)
            else:
                continue
            ret_list.append((question, answer))
    return ret_list

def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = _strip_string(str1)
        ss2 = _strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except:
        return str1 == str2

def extract_math_answer(pred_str):
    pred_str = str(pred_str or "")
    pred = ""
    if "####" in pred_str:
        pred = pred_str.rsplit("####", 1)[-1].strip()
    if not pred and 'The answer is ' in pred_str:
        pred = pred_str.split('The answer is ')[-1].strip()
    elif not pred and 'the answer is ' in pred_str:
        pred = pred_str.split('the answer is ')[-1].strip()
    elif not pred and 'boxed' in pred_str:
        ans = pred_str.split('boxed')[-1]
        if len(ans) == 0:
            print(pred_str)
        if (ans[0] == '{'):
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
        a = _strip_string(a)
        pred=a

    elif not pred:
        pattern = '-?\d*\.?\d+'
        matches = re.findall(pattern, pred_str)
        if len(matches) >= 1:
            pred = matches[-1]
        else:
            pred = ''
    if pred:
        if pred[-1] == ".":
            pred = pred[:-1]
        if pred and pred[-1] == "/":
            pred = pred[:-1]
    pred = _strip_string(pred)
    if pred and 'boxed' in pred:
        ans = pred.split('boxed')[-1]
        if not ans:
            return pred
        if (ans[0] == '{'):
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
        a = _strip_string(a)
        pred=a
    return pred

@backoff.on_exception(backoff.expo, (RateLimitError, APIError, ServiceUnavailableError, APIConnectionError, Timeout), max_tries=20)
def generate_answer(answer_context, model):
    try:
        if OPENAI_V1:
            client = openai.OpenAI(
                api_key=openai.api_key,
                base_url=openai.api_base,
            )
            completion = client.chat.completions.create(
                model=model,
                messages=answer_context,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                n=1,
            )
        else:
            completion = openai.ChatCompletion.create(
                model=model,
                messages=answer_context,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                n=1,
            )
    except RateLimitError as e:
        error_message = getattr(e, "user_message", str(e))
        if "You exceeded your current quota, please check your plan and billing details" in error_message:
            raise OutOfQuotaException(openai.api_key)
        elif "Your access was terminated due to violation of our policies" in error_message:
            raise AccessTerminatedException(openai.api_key)
        else:
            raise e

    if OPENAI_V1:
        msg = completion.choices[0].message
        model_extra = getattr(msg, "model_extra", None) or {}
        content = (
            msg.content
            or getattr(msg, "reasoning_content", None)
            or model_extra.get("reasoning_content")
            or model_extra.get("reasoning")
            or ""
        )
        return content, completion.usage.prompt_tokens, completion.usage.completion_tokens

    msg = completion["choices"][0]["message"]
    content = (
        msg.get("content")
        or msg.get("reasoning_content")
        or msg.get("reasoning")
        or ""
    )
    return content, completion["usage"]["prompt_tokens"], completion["usage"]["completion_tokens"]

def parse_single_choice(reply):
    if not reply:
        return None
    pattern = r'\(([ABCDabcd])\)'
    matches = re.findall(pattern, reply)

    solution = None
    for match_str in matches[::-1]:
        solution = match_str.upper()
        if solution:
            break

    if solution is None:
        alter_pattern = r'([ABCDabcd])\)'
        alter_matches = re.findall(alter_pattern, reply)
        for match_str in alter_matches[::-1]:
            solution = match_str.upper()
            if solution:
                break

    return solution

def most_frequent(clist, cmp_func):
    counter = 0
    num = clist[0]

    for i in clist:
        current_frequency = sum(cmp_func(i, item) for item in clist)
        if current_frequency > counter:
            counter = current_frequency
            num = i

    return num, counter
