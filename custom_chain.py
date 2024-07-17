from typing import Any, List, Mapping, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.utils import enforce_stop_tokens
from langchain.llms.base import LLM

import sys
import re

from transformers import AutoTokenizer, AutoModelForCausalLM


class CustomChain(LLM):
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None

    def __init__(self, llm=None, tokenizer=None, model=None):
        super(CustomChain, self).__init__()
        if llm is None:
            self.tokenizer = tokenizer
            self.model = model
        else:
            self.llm = llm


    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        temperature: float = 0.6,
        top_p: float = 0.9,
    ) -> str:

        #   Huggingface call.
        model_inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")
        generated_ids = self.model.generate(**model_inputs, max_new_tokens=256, temperature=0, top_p=1, max_time=50)
        results = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        results = results.replace(prompt, "")

        # print("\n\nPROMPT")
        # print(prompt)

        results = results.strip()

        if stop is not None:
            results = enforce_stop_tokens(results, stop)

        return results


    @property
    def _llm_type(self) -> str:
        return "custom"