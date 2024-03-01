import pdb

import argparse
import torch.nn.functional
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoConfig
from transformers import T5ForConditionalGeneration, PegasusForConditionalGeneration
import numpy as np
from torch.multiprocessing import Pool, Process, set_start_method
from copy import deepcopy
from bottle import Bottle, request, response, run, static_file

import json
from transformers import BitsAndBytesConfig
from peft import PeftModel
from peft import PeftConfig

from google_scripts.predict import GooglePredictor
from openai_scripts.predict import OpenaiPredictor


def make_prompt(dp, model_name):
    if "falcon" in model_name or "llama" in model_name:
        PROMPT_STR = "Generate a summary for the following document in brief. When creating the summary, only use information that is present in the document."
        input_string = f"{PROMPT_STR} DOCUMENT: {dp['document']}"
        input_string = f"{input_string} SUMMARY:"
        dp["input_string"] = input_string
    elif "flan-" in model_name:
        PROMPT_STR = "Generate a summary for the following document in brief. When creating the summary, only use information that is present in the document."
        input_string = f"{PROMPT_STR} DOCUMENT: {dp['document']}"
        dp["input_string"] = input_string
    elif "pegasus" in model_name:
        dp["input_string"] = dp['document']
    elif "mistral" in model_name:
        PROMPT_STR = "Generate a summary for the following document in brief. When creating the summary, only use information that is present in the document."
        input_string = f"{PROMPT_STR} DOCUMENT: {dp['document']}"
        dp["input_string"] = "[INST] "+input_string.strip()+" [/INST]"
    elif "gemini" in model_name or "openai" in model_name:
        PROMPT_STR = "Generate a summary for the following document in brief. When creating the summary, only use information that is present in the document. Generate the summary in freeform text without using bullet points."
        input_string = f"{PROMPT_STR} DOCUMENT: {dp['document']}"
        dp["input_string"] = input_string
    else:
        raise NotImplementedError

    return dp


def predict_generation(dp, model: AutoModelForCausalLM, tokenizer, nbeams, max_input_len, max_decode_len, do_sample=False, temperature=1.0, top_p=None, random_seed=1729):
    inputs = tokenizer(dp["input_string"], return_tensors="pt", truncation=True, max_length=max_input_len)
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)


    # pdb.set_trace()
    if do_sample:
        assert nbeams==1
        torch.random.manual_seed(seed=random_seed)

    if type(temperature)!=float:
        temperature = float(temperature)

    print("top_p=", top_p)

    gen_output = model.generate(inputs=input_ids,
                                attention_mask = attention_mask,
                                return_dict_in_generate=True,
                                output_scores=False,
                                max_length=input_ids.shape[-1]+max_decode_len,          # have to set again :( cant read from saved model
                                num_beams=nbeams,
                                top_p=top_p,
                                do_sample=do_sample,
                                temperature=temperature,
                                )
    gen_tokids = gen_output["sequences"][0]
    # pdb.set_trace()

    is_encoder_decoder = model.config.is_encoder_decoder

    if not is_encoder_decoder: # trim off the input prompt from the whole decoding
        old_numtoks = input_ids.shape[-1]
        gen_tokids = gen_tokids[old_numtoks:]

    # pdb.set_trace()

    if gen_tokids[-1].item()==tokenizer.eos_token_id:
        gen_tokids = gen_tokids[:-1]

    gen_string = tokenizer.decode(gen_tokids)

    if type(model)==T5ForConditionalGeneration or type(model)==PegasusForConditionalGeneration:
        gen_string= gen_string.lstrip("<pad>")
        gen_string = gen_string.rstrip("</s>")

    gen_string = gen_string.strip()
    print(gen_string)
    dp["prediction"] = gen_string



class HFPredictor(object):
    def __init__(self, gpu_indices, model_path, quantize, is_lora):
        if is_lora:
            adapter_config = PeftConfig.from_pretrained(model_path)
            model_name = adapter_config.base_model_name_or_path
        else:
            model_name = model_path

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False, #todo: enable this?
            trust_remote_code=True
        )

        if tokenizer.pad_token==None:
            tokenizer.pad_token = tokenizer.eos_token


        is_encoder_decoder = False
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        if config.is_encoder_decoder:
            is_encoder_decoder = True

        if  is_encoder_decoder:
            model_cls = AutoModelForSeq2SeqLM
        else:
            model_cls = AutoModelForCausalLM


        if quantize=="4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )

            model = model_cls.from_pretrained(
                model_name,
                trust_remote_code=True,
                quantization_config=bnb_config,
                device_map="auto",
                max_memory={gpu_idx: "49GB" for gpu_idx in gpu_indices}
            )

        elif quantize=="8bit":
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                # bnb_4bit_use_double_quant=True,
                # bnb_4bit_quant_type="nf4",
                # bnb_4bit_compute_dtype=torch.bfloat16
            )

            model = model_cls.from_pretrained(
                model_name,
                trust_remote_code=True,
                quantization_config=bnb_config,
                device_map="auto",
                max_memory={gpu_idx: "49GB" for gpu_idx in gpu_indices}
            )



        elif quantize=="16bit":
            print("USING 16BIT INFERENCE")
            model = model_cls.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                max_memory={gpu_idx: "49GB" for gpu_idx in gpu_indices}
            )
            # assert len(gpu_indices)==1
            # gpu_idx = gpu_indices[0]
            # model = model.to(torch.device(f"cuda:{gpu_idx}"))

        else:
            print("quanitzation type ", quantize, "not supported!")
            raise NotImplementedError


        # good for making generation fast
        model.config.use_cache=True


        if is_lora:
            model = PeftModel.from_pretrained(model,
                                             model_path,
                                             torch_dtype=torch.bfloat16,
                                             device_map="auto",
                                             max_memory={gpu_idx: "49GB" for gpu_idx in gpu_indices})

            # inexplicably the above code sometimes does not load model to gpu even though we have specified the device map
            # this happened with bf16 + lora + flant5; on the other hand 4bit + lora + flan/mistral/llama loads the model correctly into gpu
            # so here we manually check and move the model to gpu if it isnt there
            if model.device==torch.device("cpu"):
                model = model.cuda()

        # pdb.set_trace()

        # predict_generation({"input_string":"The capital of Pennsylvania is"}, model, tokenizer, 1, 9999, 5)

        # for tval in [1.0, 99, 999, 9999, 99999, 999999]:
        for tval in [1.1]:
            predict_generation(dp={"input_string":"One thing to do to cure a hangover is to"},
                               model=model,
                               tokenizer=tokenizer,
                               nbeams=1,
                               max_input_len=9999,
                               max_decode_len=5,
                               do_sample=True,
                               temperature=tval)

        self.model = model
        self.tokenizer = tokenizer


    def predict(self, dp, max_input_len, max_decode_len, temperature, dosample, top_p):
        toklen = len(self.tokenizer.encode(dp["input_string"]))
        if toklen>max_input_len:
            return {"result": None, "success": False, "reason": "length overflow"}

        predict_generation(dp, model=self.model, tokenizer=self.tokenizer, nbeams=1, max_input_len=max_input_len, max_decode_len=max_decode_len, temperature=temperature, do_sample=dosample, top_p=top_p)

        # at this point the dp are updated in place with results

        if "prediction" not in dp:
            return {"result": None, "success": False, "reason": "unknown"}

        return {"result": dp["prediction"], "success": True}


class GAIPredictor(object):
    def __init__(self, model_name):
        self.api_caller = GooglePredictor(model_name=model_name)

    def predict(self, dp, max_decode_len, **kwargs):
        msglist = [{"role":"user", "parts":[dp["input_string"]]}]
        final_dp = {"input_messages": msglist, "max_decode_len":max_decode_len}
        self.api_caller.predict(final_dp)

        if "prediction" in final_dp:
            print("GOOGLE API RETURNED : ", final_dp["prediction"])
            return {"result": final_dp["prediction"], "success": True}
        else:
            return {"result": None, "success": False, "reason": "unknown"}


class OAIPredictor(object):
    def __init__(self, model_name):
        self.api_caller = OpenaiPredictor(model_name=model_name)

    def predict(self, dp, max_decode_len, **kwargs):
        msglist = [{"role":"user", "content":dp["input_string"]}]
        final_dp = {"input_messages": msglist, "max_decode_len":max_decode_len}
        self.api_caller.predict(final_dp)

        if "prediction" in final_dp:
            return {"result": final_dp["prediction"], "success": True}
        else:
            return {"result": None, "success": False, "reason": "unknown"}


def process_root(pidx, gpu_indices, model_path, port_val, quantize, is_lora, max_input_len, max_decode_len):

    if "google:"==model_path[:7]:
        google_model_name = model_path.split("google:")[-1]
        process_predictor = GAIPredictor(model_name=google_model_name)
    elif "openai:"==model_path[:7]:
        oai_model_name = model_path.split("openai:")[-1]
        process_predictor = OAIPredictor(model_name=oai_model_name)
    else:
        process_predictor = HFPredictor(gpu_indices=gpu_indices, model_path=model_path, quantize=quantize, is_lora=is_lora)

    app = Bottle()


    @app.hook('after_request')
    def enable_cors():
        """
        You need to add some headers to each request.
        Don't use the wildcard '*' for Access-Control-Allow-Origin in production.
        """
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'PUT, GET, POST, DELETE, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Origin, Accept, Content-Type, X-Requested-With, X-CSRF-Token'


    @app.route("/get_config", method=['GET'])
    def get_config():
        return  {
            "model_name_or_path": model_path,
            "max_input_len": max_input_len,
            "max_decode_len": max_decode_len,
            "precision": quantize,
            "beam_width": 1
        }

    @app.route("/test", method=["GET"])
    def test():
        return {"success": True}


    @app.route('/predict', method=['POST'])
    def predict():
        dp = request.json

        temperature = dp["temperature"]
        top_p = dp["top_p"]
        dosample = dp["dosample"]

        newdp = make_prompt(dp, model_path)

        # print(newdp)

        return process_predictor.predict(dp=newdp,
                                         max_input_len=max_input_len,
                                         max_decode_len=max_decode_len,
                                         temperature=temperature,
                                         dosample=dosample,
                                         top_p=top_p)


    run(app, host="localhost", port=port_val, debug=True)


if __name__=="__main__":

    np.random.seed(1337)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--base-port", type=int, default=9000)
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--quantize", type=str, default="16bit")
    parser.add_argument("--max-input-len", type=int, default=1000)
    parser.add_argument("--max-decode-len", type=int, default=150)
    parser.add_argument("--is-lora", action="store_true")
    args = parser.parse_args()

    model_name = args.model_name
    base_port = args.base_port
    quantize = args.quantize
    max_input_len = args.max_input_len
    max_decode_len = args.max_decode_len
    num_processes = args.num_processes
    is_lora = args.is_lora

    set_start_method("spawn")

    all_procs = []
    for pidx in range(num_processes):
        p = Process(target=process_root, args=(pidx, [pidx], model_name, base_port, quantize, is_lora, max_input_len, max_decode_len))
        p.start()
        all_procs.append(p)

    for p in all_procs:
        p.join()
