import os
import pdb

import jsonlines
import tiktoken
from tqdm import tqdm
import numpy as np
import argparse

import google.generativeai as genai

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff


SAFETY_SETTINGS = [
      {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_NONE"
      },
      {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_NONE"
      },
      {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_NONE"
      },
      {
    "category": "HARM_CATEGORY_DANGEROUS",
    "threshold": "BLOCK_NONE"
      },
  ]


class GooglePredictor(object):
    def __init__(self, model_name):
        GOOGLE_API_KEY = ""
        genai.configure(api_key=GOOGLE_API_KEY)
        self.model_name = model_name
        self.client = genai.GenerativeModel(model_name)

    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(6))
    def get_response(self,msg_list, max_tokens=None):
        try:
            response=self.client.generate_content(msg_list, safety_settings=SAFETY_SETTINGS, generation_config={"max_output_tokens": max_tokens})
            return response.text
        except:
            print(response.prompt_feedback)
            return None

    def predict(self, dp):
        pred_str = self.get_response(dp["input_messages"], dp["max_decode_len"])
        if pred_str!=None:
            dp["prediction"] = pred_str



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="gemini-pro")
    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file
    model_name = args.model_name

    google_predictor = GooglePredictor(model_name=model_name)

    with jsonlines.open(output_file, "w") as w:
        for dp in tqdm(jsonlines.open(input_file,"r")):
            google_predictor.predict(dp)
            if "prediction" not in dp:
                pdb.set_trace()
            w.write(dp)


