import json
import difflib
from pydantic import BaseModel, Field, TypeAdapter
from openai import OpenAI
from copy import deepcopy
from typing import Union, Optional, List, Dict, Any, Tuple, Literal
import re
import ast
from .prompts.filter_default_prompt import best_dspy_prompt

class Fact(BaseModel):
    fact: list[list[str]] = Field(description="A list of facts, each fact is a list of 3 strings: [subject, predicate, object]")


class DSPyFilter:
    def __init__(self, srgraphrag):
        dspy_file_path = srgraphrag.global_config.rerank_dspy_file_path
        self.one_input_template = (
            "[[ ## question ## ]]\n{question}\n\n"
            "[[ ## fact_before_filter ## ]]\n{fact_before_filter}\n\n"
            "Respond with the corresponding output fields, starting with the field "
            "`[[ ## fact_after_filter ## ]]` (must be formatted as a valid Python Fact), "
            "and then ending with the marker for `[[ ## completed ## ]]`."
        )
        self.one_output_template = (
            "[[ ## fact_after_filter ## ]]\n{fact_after_filter}\n\n"
            "[[ ## completed ## ]]"
        )
        self.message_template = self.make_template(dspy_file_path)
        self.llm_infer_fn = srgraphrag.llm_model.infer
        self.model_name = srgraphrag.global_config.llm_name
        self.default_gen_kwargs = {}

    def make_template(self, dspy_file_path):
        if dspy_file_path is not None:
            dspy_saved = json.load(open(dspy_file_path, "r"))
        else:
            dspy_saved = best_dspy_prompt

        system_prompt = dspy_saved["prog"]["system"]
        message_template = [{"role": "system", "content": system_prompt}]

        demos = dspy_saved["prog"].get("demos", [])
        for demo in demos:
            message_template.append({
                "role": "user",
                "content": self.one_input_template.format(
                    question=demo["question"],
                    fact_before_filter=demo["fact_before_filter"]
                )
            })
            message_template.append({
                "role": "assistant",
                "content": self.one_output_template.format(
                    fact_after_filter=demo["fact_after_filter"]
                )
            })
        return message_template

    def _to_text(self, response: Any) -> str:
        """
        Make response always a string so parse_filter won't crash.
        Handles:
        - str
        - list/tuple (take first element)
        - dict with common fields
        - other -> str()
        """
        if response is None:
            return ""

        # already string
        if isinstance(response, str):
            return response

        # list/tuple: take first element (often model returns [text] or (text, meta))
        if isinstance(response, (list, tuple)):
            if len(response) == 0:
                return ""
            return self._to_text(response[0])

        # dict: try typical keys
        if isinstance(response, dict):
            for key in ["content", "text", "output", "message"]:
                if key in response:
                    return self._to_text(response[key])
            return str(response)

        return str(response)

    def parse_filter(self, response: Any):

        response_text = self._to_text(response)

        sections = [(None, [])]
        field_header_pattern = re.compile(r"\[\[ ## (\w+) ## \]\]")

        for line in response_text.splitlines():
            match = field_header_pattern.match(line.strip())
            if match:
                sections.append((match.group(1), []))
            else:
                sections[-1][1].append(line)

        sections = [(k, "\n".join(v).strip()) for k, v in sections]

        parsed = []
        for k, value in sections:
            if k == "fact_after_filter":
                try:
                    # 1) try json
                    try:
                        parsed_value = json.loads(value)
                    except json.JSONDecodeError:
                        # 2) try python literal
                        try:
                            parsed_value = ast.literal_eval(value)
                        except (ValueError, SyntaxError):
                            parsed_value = value

                    # 3) if not proper dict, try extract triples from text
                    if not (isinstance(parsed_value, dict) and "fact" in parsed_value):
                        triples = re.findall(
                            r'\[\s*"([^"]*)"\s*,\s*"([^"]*)"\s*,\s*"([^"]*)"\s*\]',
                            value
                        )
                        if triples:
                            parsed_value = {"fact": [list(t) for t in triples]}
                        else:
                            return []

                    # validate -> Fact -> list of triples
                    parsed = TypeAdapter(Fact).validate_python(parsed_value).fact
                except Exception as e:
                    print(
                        f"Error parsing field {k}: {e}.\n\nOn attempting to parse the value\n```\n{value}\n```"
                    )

        return parsed

    def llm_call(self, question, fact_before_filter, instruction: str | None = None):
        messages = deepcopy(self.message_template)

        user_content = self.one_input_template.format(
            question=question,
            fact_before_filter=fact_before_filter
        )

        if instruction:
            user_content = instruction.strip() + "\n\n" + user_content

        messages.append({"role": "user", "content": user_content})

        self.default_gen_kwargs["max_completion_tokens"] = 1024

        response = self.llm_infer_fn(
            messages=messages,
            model=self.model_name,
            **self.default_gen_kwargs
        )

        return response

    def __call__(self, *args, **kwargs):
        return self.rerank(*args, **kwargs)

    def rerank(
        self,
        query: str,
        candidate_items: List[Tuple],
        candidate_indices: List[int],
        len_after_rerank: int = None,
        instruction: str | None = None,  
    ) -> Tuple[List[int], List[Tuple], dict]:

        fact_before_filter = {"fact": [list(candidate_item) for candidate_item in candidate_items]}

        try:
            response = self.llm_call(query, json.dumps(fact_before_filter), instruction=instruction)
            generated_facts = self.parse_filter(response)
        except Exception as e:
            print("exception", e)
            generated_facts = []

        result_indices = []
        cand_strs = [str(i) for i in candidate_items]

        for generated_fact in generated_facts:
            matches = difflib.get_close_matches(
                str(generated_fact),
                cand_strs,
                n=1,
                cutoff=0.0
            )
            if not matches:
                continue
            closest_matched_fact = matches[0]

            try:
                result_indices.append(candidate_items.index(eval(closest_matched_fact)))
            except Exception as e:
                print("result_indices exception", e)

        sorted_candidate_indices = [candidate_indices[i] for i in result_indices]
        sorted_candidate_items = [candidate_items[i] for i in result_indices]

        if len_after_rerank is None:
            return sorted_candidate_indices, sorted_candidate_items, {"confidence": None}

        return (
            sorted_candidate_indices[:len_after_rerank],
            sorted_candidate_items[:len_after_rerank],
            {"confidence": None}
        )