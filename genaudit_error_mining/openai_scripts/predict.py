import os
import pdb

import jsonlines
import tiktoken
from tqdm import tqdm
import numpy as np
import argparse

from openai import OpenAI

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff


class OpenaiPredictor(object):
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = tiktoken.encoding_for_model(model_name)
        self.client = OpenAI(api_key = "")
    def get_approx_promptlen(self,msgs):
        total_len = 0
        for obj in msgs:
            total_len += len(self.tokenizer.encode(obj["content"]))
        return total_len

    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(6))
    def get_response(self,msg_list, max_tokens=None):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=msg_list,
            max_tokens=max_tokens
                )
        return response.choices[0].message.content

    def predict(self, dp):
        final_payload = dp["input_messages"]
        max_tokens = dp["max_decode_len"]
        num_toks = self.get_approx_promptlen(final_payload)

        if self.model_name=="gpt-3.5-turbo-16k-0613" and num_toks>16000:
            # probably too long
            pdb.set_trace()

        if self.model_name=="gpt-4-0613" and num_toks>8000:
            # probably too long
            pdb.set_trace()

        dp["prediction"] = self.get_response(final_payload, max_tokens)



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="gpt-3.5-turbo-16k-0613")
    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file
    model_name = args.model_name

    openai_predictor = OpenaiPredictor(model_name=model_name)

    with jsonlines.open(output_file, "w") as w:
        for dp in tqdm(jsonlines.open(input_file,"r")):
            dp["max_decode_len"] = None # dont enforce a max decode len
            openai_predictor.predict(dp)
            if "prediction" not in dp:
                pdb.set_trace()
            w.write(dp)


