"""Subject-budget prompt and candidate statistics used by the PPR baseline."""

from __future__ import annotations

import re


def _cap_to_word(n: int) -> str:
    m = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
    return m.get(n, "four")


def build_subject_cap_instruction(n: int) -> str:
    n = int(n)
    n = max(1, min(5, n))

    base = (
        "You are a critical component of a high-stakes question-answering system used by top researchers "
        "and decision-makers worldwide. Your task is to filter facts based on their relevance to a given query, "
        "ensuring that the most crucial information is presented to these stakeholders. The query requires careful "
        "analysis and possibly multi-hop reasoning to connect different pieces of information. "
        "You must select all facts from the provided candidate list that are strongly relevant to the query, "
        "prioritizing at most four key subjects and including as many relevant facts as possible under those subjects, "
        "while ensuring that the total number of unique subjects among the selected facts does not exceed four, "
        "aiding in reasoning and providing an accurate answer. "
        'The output should be in JSON format, e.g., {"fact": [["s1", "p1", "o1"], ["s2", "p2", "o2"]]}, '
        'and if no facts are relevant, return an empty list, {"fact": []}. '
        "The accuracy of your response is paramount, as it will directly impact the decisions made by these high-level stakeholders. "
        "You must only use facts from the candidate list and not generate new facts."
    )
    word = _cap_to_word(n)
    out = re.sub(r"\bfour\b", word, base)
    out = re.sub(r"\b4\b", str(n), out)
    return out


def count_unique_subjects(triples):
    if not triples:
        return 0
    return len({t[0] for t in triples if isinstance(t, (list, tuple)) and len(t) >= 3})
