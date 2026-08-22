import glob
import json
import re
import pandas as pd
from typing import Union, List, Literal, Any, Dict
import numpy as np
from abc import ABC

def _strip_reasoning_blocks(text: str) -> str:
    t = text.strip().replace("\r\n", "\n")
    _think_open = "`" * 3 + "think" + "\n"
    _think_close = "\n" + "`" * 3
    if _think_open in t and _think_close in t:
        t = t.split(_think_close, 1)[-1].strip()
    elif _think_open in t:
        t = t.split(_think_open, 1)[-1].strip()
    if "<think>" in t and "</think>" in t:
        t = t.split("</think>", 1)[-1].strip()
    elif "</think>" in t:
        t = t.split("</think>")[-1].strip()
    return t

def _extract_choice(raw: str) -> str:
    if not raw:
        return ""
    t = _strip_reasoning_blocks(raw)
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if lines:
        last = lines[-1].strip()
        if last.startswith("{") and "answer" in last.lower():
            try:
                obj = json.loads(last)
                for key in ("answer", "choice", "letter"):
                    v = obj.get(key)
                    if isinstance(v, str) and v.upper().strip()[:1] in "ABCD":
                        return v.upper().strip()[:1]
            except Exception:
                pass
    for ln in reversed(lines[-8:]):
        u = ln.strip()
        m = re.match(r"^[\[(]*([ABCD])[)\].\s:：]*$", u, re.I)
        if m:
            return m.group(1).upper()
        m = re.search(r"(?i)(?:answer|choice|option)\s*[:：]?\s*([ABCD])\b", u)
        if m:
            return m.group(1).upper()
    phrase_patterns = [
        r"(?i)answer\s*is\s*:?\s*([ABCD])\b",
        r"(?i)correct\s*(?:answer|option)\s*is\s*:?\s*([ABCD])\b",
        r"(?i)therefore[, ]+\s*(?:option\s*)?([ABCD])\b",
        r"(?i)I\s+(?:would\s+)?(?:choose|select|pick)\s+([ABCD])\b",
    ]
    for pat in phrase_patterns:
        found = re.findall(pat, t)
        if found:
            return str(found[-1]).upper()
    letters = re.findall(r"\b([ABCD])\b", t.upper())
    if letters:
        return letters[-1]
    u = t.upper().strip()
    if len(u) == 1 and u in "ABCD":
        return u
    return ""

class MMLUDataset(ABC):
    def __init__(self,
        split: Union[Literal['dev'], Literal['val'], Literal['test']],
        ) -> None:

        self._split = split

        data_path = f"datasets/MMLU/data/{self._split}/"
        self._total_df: pd.DataFrame = self._load_data(data_path)

    @staticmethod
    def get_domain() -> str:
        return 'mmlu'

    @staticmethod
    def _load_data(
        data_path: str,
        ) -> pd.DataFrame:

        rng = np.random.default_rng(888)

        csv_paths = glob.glob(data_path + "*.csv")
        csv_paths = sorted(csv_paths)
        print("Number of topics: ", len(csv_paths))

        names = ['question', 'A', 'B', 'C', 'D', 'correct_answer']

        total_df = pd.DataFrame(columns=names)
        for path in csv_paths:
            single_df = pd.read_csv(path, header=None,
                            names=names,encoding='utf-8')
            total_df = pd.concat([total_df, single_df])

        total_df = total_df.reset_index(drop=True)


        total_df = total_df.reindex(rng.permutation(total_df.index))

        print("Total number of questions: ", len(total_df))

        return total_df

    @property
    def split(self) -> str:
        return self._split

    def __len__(self) -> int:
        return len(self._total_df)

    def __getitem__(self, index: int) -> pd.DataFrame:
        record = self._total_df.iloc[index]
        assert isinstance(record, pd.DataFrame) or isinstance(record, pd.Series)
        return record

    @staticmethod
    def record_to_input(record: pd.DataFrame) -> Dict[str, Any]:
        demo_question = (
            f"{record['question']}\n"
            f"Option A: {record['A']}\n"
            f"Option B: {record['B']}\n"
            f"Option C: {record['C']}\n"
            f"Option D: {record['D']}\n"
            )
        input_dict = {"task": demo_question}
        return input_dict

    def postprocess_answer(self, answer: Union[str, List[str]]) -> str:
        if isinstance(answer, list):
            answer = answer[0] if len(answer) > 0 else ""
        if not isinstance(answer, str):
            raise Exception("Expected string")
        letter = _extract_choice(answer)
        if letter:
            return letter
        if len(answer) > 0:
            ans_pos = answer.find("answer is")
            if ans_pos != -1:
                tail = answer[ans_pos + len("answer is"):].strip(":").strip().strip("Option").strip()
                if tail:
                    return tail[0]
            return answer.strip()[:1]
        return ""

    @staticmethod
    def record_to_target_answer(record: pd.DataFrame) -> str:
        correct_answer = record['correct_answer']
        assert isinstance(correct_answer, str), (
            f"String expected but got {correct_answer} "
            f"of type {type(correct_answer)} (2)"            f" record={record}")
        return correct_answer
