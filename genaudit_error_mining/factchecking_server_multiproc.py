import argparse
import difflib
import json
import multiprocessing
import pdb
from collections import defaultdict
from copy import deepcopy

import jsonlines
import nltk
import requests
import torch
from tqdm import tqdm
from datasets import load_dataset
from nltk.tokenize import TreebankWordTokenizer as twt
from transformers import AutoTokenizer

from bottle import Bottle, request, response, run, static_file

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import BitsAndBytesConfig
from peft import PeftModel
from peft import PeftConfig

from torch.multiprocessing import Process, set_start_method

from openai_scripts.predict import OpenaiPredictor


def process_evidence_extraction_with_fixfactuality_prependsumm(dp):
    PROMPT_STR = "You are provided a document and its summary. The summary may potentially contain factual errors. The last sentence of the summary is marked as a claim. Find all sentences in the document providing evidence for the claim, and then revise the claim to remove or replace unsupported facts."
    input_string = f"{PROMPT_STR} DOCUMENT:"
    for _i,sent in enumerate(dp["input_lines"]):
        input_string = f"{input_string} SENT{_i} {sent}"

    input_string = f"{input_string} SUMMARY:"
    for _k, sent in enumerate(dp["prev_summ_lines"]):
        input_string = f"{input_string} {sent}"

    input_string = f"{input_string} CLAIM: {dp['before_summary_sent']}"

    output_string = f"EVIDENCE:"
    for ev_idx in dp["evidence_labels"]:
        output_string = f"{output_string} SENT{ev_idx}"
    output_string = f"{output_string} REVISION: {dp['after_summary_sent']}"

    dp["input_string"] = input_string
    dp["output_string"] = output_string

    return dp




def predict_generation(dp, model: AutoModelForSeq2SeqLM, tokenizer, nbeams, max_input_len, max_decode_len):
    inputs = tokenizer(dp["input_string"], return_tensors="pt", truncation=True, max_length=max_input_len)
    input_ids = inputs.input_ids.cuda()

    gen_output = model.generate(inputs=input_ids,
                                return_dict_in_generate=True,
                                decoder_input_ids=None,
                                output_scores=False,
                                max_length=max_decode_len,
                                num_beams=nbeams)

    gen_tokids = gen_output["sequences"][0]

    gen_tokids = gen_tokids[1:] # first token is pad
    if gen_tokids[-1].item()==tokenizer.eos_token_id:
        gen_tokids = gen_tokids[:-1]

    if "before_summary_sent" in dp.keys():
        print("INPUT was:", dp["before_summary_sent"])

    gen_string = tokenizer.decode(gen_tokids)
    print("FC generated: ", gen_string)
    dp["prediction"] = gen_string



def showdiff(fr, to):
    differ = difflib.Differ()

    fr_wordspans = list(twt().span_tokenize(fr))
    to_wordspans = list(twt().span_tokenize(to))

    fr_words = []
    last_endpos = 0
    for onespan in fr_wordspans:
        fr_words.append(fr[last_endpos:onespan[1]])
        last_endpos = onespan[1]

    to_words = []
    last_endpos = 0
    for onespan in to_wordspans:
        to_words.append(to[last_endpos:onespan[1]])
        last_endpos = onespan[1]

    line = ""
    deleteonly_line = ""
    normal_spans = []
    deleted_spans = []
    added_spans = []
    deleteonly_spans = []
    addonly_insertionmap = defaultdict(str)

    entries = list(differ.compare(fr_words, to_words))

    entries = [e for e in entries if e[0]!="?"]

    # this loop rearranges words in the diff such that the additions come after deletions
    # this loop is guaranteed to terminate because at each iteration, we either advance one + over - (like checkers) or exit the subroutine
    # since there are limited number of -s that can be hopped over, we will terminate
    # time complexity is O(n^2) but modern computers are fast enough for this
    while True:
        swapped_something = False
        for i in range(len(entries)-1):
            if entries[i][0]=="+" and entries[i+1][0]=="-":
                #swap
                temp = entries[i]
                entries[i] = entries[i+1]
                entries[i+1] = temp
                swapped_something = True
                break
        if not swapped_something:
            break

    for entry in entries:
        if entry[0]=="+":
            text = entry[2:]
            start_idx = len(line)
            end_index = start_idx+len(text)

            # this means we are at the start of a potentially multi-word addition
            if len(deleted_spans)>0:
                if deleted_spans[-1][1]==start_idx:
                    addonly_insertionmap[deleteonly_spans[-1][1]] += text

            # this means we are at word_idx>=1 of a multi-word addition. so just expand the last addition.
            if len(added_spans)>0:
                if added_spans[-1][1]==start_idx:
                    addonly_insertionmap[deleteonly_spans[-1][1]] += text
            else:
                pass

            added_spans.append((start_idx, end_index))
            line+=text



        elif entry[0]=="-":
            text = entry[2:]
            start_idx = len(line)
            end_index = start_idx+len(text)
            deleted_spans.append((start_idx, end_index))
            line+=text

            start_idx = len(deleteonly_line)
            end_index = start_idx+len(text)
            deleteonly_spans.append((start_idx, end_index))
            deleteonly_line+=text

        else:
            text = entry[2:]
            start_idx = len(line)
            end_index = start_idx+len(text)
            normal_spans.append((start_idx, end_index))
            line+=text

            start_idx = len(deleteonly_line)
            end_index = start_idx+len(text)
            deleteonly_line+=text

    return {
        "line": line,
        "normal_spans": normal_spans,
        "deleted_spans": deleted_spans,
        "added_spans": added_spans,
        "deleteonly_line": deleteonly_line,
        "deleteonly_spans": deleteonly_spans,
        "addonly_insertionmap": addonly_insertionmap
    }


class HFPredictor(object):
    def __init__(self, gpu_idx, model_name,  max_input_len,  max_decode_len):

        adapter_config = PeftConfig.from_pretrained(model_name)
        base_model_name_or_path = adapter_config.base_model_name_or_path

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name_or_path,
            use_fast=False,
            trust_remote_code=True
        )

        if tokenizer.pad_token==None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForSeq2SeqLM.from_pretrained(
            base_model_name_or_path,
            trust_remote_code=True,
            quantization_config=bnb_config,
            device_map={'':gpu_idx}
        )
        # good for making generation fast
        model.config.use_cache=True

        mdl2 = PeftModel.from_pretrained(model,
                                         model_name,
                                         torch_dtype=torch.bfloat16,
                                         device_map={'':gpu_idx})

        model.gradient_checkpointing_disable()
        mdl2.gradient_checkpointing_disable()

        # test run
        predict_generation({"input_string":"The capital of Pennsylvania is"}, mdl2, tokenizer, 1, 9999, 5)

        self.mdl2 = mdl2
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_decode_len = max_decode_len


    def get_evidence_extraction_with_fixfactuality(self, input_dict):
        newdp = process_evidence_extraction_with_fixfactuality_prependsumm(input_dict)
        predict_generation(newdp, model=self.mdl2, tokenizer=self.tokenizer, nbeams=4, max_input_len=self.max_input_len, max_decode_len=self.max_decode_len)
        return newdp



class OAIPredictor(object):
    def __init__(self, model_name):
        self.api_caller = OpenaiPredictor(model_name=model_name)
        # use 4 or 8 shots depending on context length
        if model_name=="gpt-3.5-turbo-16k-0613":
            prefix_msgpath = "/home/kundank/dataset_root/factual_editor/for_common_inference/all_0.03_0.03/for_gpts/gpt-3.5-turbo-16k-0613/evidence_extraction_with_fixfactuality_prependsumm_nofulldelete_maxinplen_3000_maxoutlen_100/test.jsonl"
        elif model_name=="gpt-4-0613":
            prefix_msgpath = "/home/kundank/dataset_root/factual_editor/for_common_inference/all_0.03_0.03/for_gpts/gpt-4-0613/evidence_extraction_with_fixfactuality_prependsumm_nofulldelete_maxinplen_3000_maxoutlen_100/test.jsonl"
        else:
            raise NotImplementedError
        self.prefix_messages = list(jsonlines.open(prefix_msgpath))[0]["input_messages"][:-1]


    def get_evidence_extraction_with_fixfactuality(self, input_dict):
        newdp = process_evidence_extraction_with_fixfactuality_prependsumm(input_dict)
        inpsplit = newdp["input_string"].split("DOCUMENT: ")
        assert len(inpsplit)==2
        inp = f"DOCUMENT: {inpsplit[1]}"
        final_dp = {
            "input_messages" : deepcopy(self.prefix_messages) + [{"role":"user", "content": inp}],
            "max_decode_len": None
        }
        self.api_caller.predict(final_dp)

        if "prediction" not in final_dp:
            # failed
            pdb.set_trace()

        return final_dp



def process_root(pidx, gpu_idx, model_name, port_val, max_input_len, max_decode_len):

    if "openai:" in model_name:
        oai_model_name = model_name.split("openai:")[-1]
        process_predictor = OAIPredictor(model_name=oai_model_name)
        pass
    else:
        process_predictor = HFPredictor(gpu_idx=gpu_idx, model_name=model_name, max_input_len=max_input_len, max_decode_len=max_decode_len)

    def predict_fix(article_lines, summary_line, prev_summ_lines):

        num_frontspaces = len(summary_line)-len(summary_line.lstrip())
        summary_line = summary_line.strip()


        ev_fixfactuality_output = process_predictor.get_evidence_extraction_with_fixfactuality({
              'input_lines': article_lines,
              'prev_summ_lines': prev_summ_lines,
              'before_summary_sent': summary_line,
              'after_summary_sent': "dummmy",
              'id': 'xxxxx',
              'evidence_labels': [0]
            })

        output = ev_fixfactuality_output["prediction"]

        try:
            fixed_output = output.split("REVISION:")[1].strip()
            ev_sentids = output.split("REVISION:")[0].split("EVIDENCE: ")[1].strip()
        except IndexError:
            # if the output isnt formatted properly (usually rare), then fall back by not suggesting any edits or evidence
            fixed_output = summary_line
            ev_sentids = ""

        # print("BEFORE:", summary_line)
        # print("AFTER:", fixed_output)

        ev_labels = []
        for one_sentid in ev_sentids.split(" "):
            try:
                this_idx = one_sentid.split("SENT")[-1]
                ev_labels.append(int(this_idx))
            except:
                # this can happen if the evidence isnt formatted properly
                print("Throwing away badly formatted evidence labels:", one_sentid)
                continue

        try:
            diff_bw_two = showdiff(fr=summary_line, to=fixed_output)
            todelete_spans = diff_bw_two["deleteonly_spans"]
            addonly_insertionmap = diff_bw_two["addonly_insertionmap"]
        except:
            # triggered if the edits made by the model cant be localized properly
            # in that case just fall back to saying no changes were predicted
            todelete_spans = []
            addonly_insertionmap = {}

        # converting from tuples to list to modify
        todelete_spans = [list(x) for x in todelete_spans]

        replacement_strings = []
        for onespan in todelete_spans:
            endpos = onespan[-1]
            if endpos in addonly_insertionmap:
                replacement_strings.append(addonly_insertionmap[endpos])
            else:
                replacement_strings.append("")


        # filter the replacements to disallow certain ones that are problematic (e.g. unks, only spaces)
        filtered_todelete_spans = []
        filtered_replacement_strings = []
        for (onespan, repstr) in zip(todelete_spans, replacement_strings):
            # if unk then skip
            if "<unk>" in repstr:
                print("FILTER ALERT: skipped replacement of ", repstr)
                continue

            # if the difference is only whitespace then skip
            l,r = onespan
            before_str = summary_line[l:r]
            if before_str.strip()==repstr.strip():
                print(f"FILTER ALERT: skipped replacement of identical except whitespace *{before_str}* *{repstr}*")
                continue

            filtered_todelete_spans.append(onespan)
            filtered_replacement_strings.append(repstr)


        todelete_spans = filtered_todelete_spans
        replacement_strings = filtered_replacement_strings


        fused_todelete_spans = []
        fused_replacement_strings = []
        for (onespan, repl) in zip(todelete_spans, replacement_strings):
            if len(fused_todelete_spans)==0 or onespan[0]!=fused_todelete_spans[-1][1]:
                fused_todelete_spans.append(onespan)
                fused_replacement_strings.append(repl)
            else:
                fused_todelete_spans[-1][1] = onespan[1]
                fused_replacement_strings[-1] += repl

        assert len(fused_todelete_spans)==len(fused_replacement_strings)

        # adjust for the spaces at the beginning
        for j in range(len(fused_todelete_spans)):
            fused_todelete_spans[j][0] += num_frontspaces
            fused_todelete_spans[j][1] += num_frontspaces

        return {"evidence_labels": ev_labels,
                "todelete_spans": fused_todelete_spans,
                "replacement_strings": fused_replacement_strings}


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
            "model_name_or_path": model_name,
            "max_input_len": max_input_len,
            "max_decode_len": max_decode_len,
            "beam_width": 1
        }

    @app.route("/test", method=["GET"])
    def test():
        return {"success": True}

    @app.route('/predict', method=['POST'])
    def predict():
        dp = request.json


        article_lines = dp["document"].strip().split("\n")

        if type(dp["summary"])==list:
            summary_lines=dp["summary"]
        elif type(dp["summary"])==str:
            gen_summary = dp["summary"].strip()
            summary_lines = nltk.sent_tokenize(gen_summary)
        else:
            raise NotImplementedError

        # length_data = check_length(input_lines=article_lines, tokenizer=tokenizer)
        #
        # print("XXXXXXXXXXXXXXXXXXXXXXX", length_data, dp)
        #
        # if not length_data["okay"]:
        #     return {"result":None, "success":False, "reason":"length too long"}

        dp["article_lines"] = article_lines
        dp["summary_lines"] = summary_lines
        dp["factcheck_predictions"] = []

        # try:
        for k,summary_line in enumerate(summary_lines):
            result = predict_fix(article_lines, summary_line, summary_lines[:k]) # last argument is previous summary lines
            dp["factcheck_predictions"].append(result)

        # except:
        #     return {"result":None, "success":False, "reason":"likely OOM error"}

        return {"result":dp, "success":True, "reason":""}


    run(app, host="localhost", port=port_val, debug=True)



if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--base-port", type=int, default=9000)
    parser.add_argument("--model-name", type=str, required=True)

    set_start_method("spawn")

    args = parser.parse_args()
    N_PROCS = args.num_processes
    base_port = args.base_port
    model_name = args.model_name

    jobs = []
    for i in range(N_PROCS):
      job = Process(target=process_root, args=(i,i, model_name, base_port, 9999999, 9999999))
      jobs.append(job)
      job.start()
    for job in jobs:
      job.join()

