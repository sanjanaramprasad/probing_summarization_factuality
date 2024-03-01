from datasets import load_dataset
import pandas as pd
from transformers import AutoTokenizer
import numpy as np
from copy import deepcopy

def truncate(article_lines, tokenizer, maxlen):
    toklens = [len(tokenizer.encode(x)) for x in article_lines]
    totalsums = np.cumsum(toklens)
    idx = 0
    while totalsums[idx]<maxlen:
        idx+=1
        if idx==len(totalsums):
            break

    # truncate
    return deepcopy(article_lines[:idx])


def get_data(dsname, do_truncate=False, maxlen=2000):
    tok = AutoTokenizer.from_pretrained("google/flan-t5-large")

    if dsname=="scitldr":
        scitldr = load_dataset("allenai/scitldr")
        test_set = [dict(x) for x in scitldr["test"]]
        for sample_dp in test_set:
            article_lines = [x.strip().replace("\n"," ") for x in sample_dp["source"]]
            if do_truncate:
                article_lines = truncate(article_lines, tok, maxlen)
            yield {"input_lines": article_lines, "id":sample_dp["paper_id"]}


    elif dsname=="summscreen":
        scrolls = load_dataset("tau/scrolls", "summ_screen_fd")
        test_set = [dict(x) for x in scrolls["test"]]
        for sample_dp in test_set:
            article_lines = sample_dp["input"].strip().split("\n")
            if do_truncate:
                article_lines = truncate(article_lines, tok, maxlen)
            yield {"input_lines": article_lines, "id":sample_dp["pid"]}

    elif dsname=="acibench":
        aci_ds = pd.read_csv("https://raw.githubusercontent.com/wyim/aci-bench/main/data/challenge_data/clinicalnlp_taskB_test1.csv")
        N=len(aci_ds)
        test_set = [dict(aci_ds.loc[idx]) for idx in range(N)]
        for sample_dp in test_set:
            article_lines = sample_dp["dialogue"].strip().split("\n")
            if do_truncate:
                article_lines = truncate(article_lines, tok, maxlen)
            yield {"input_lines": article_lines, "id":sample_dp["encounter_id"]}

    elif dsname=="xsum":
        xsum = load_dataset("xsum")
        test_set = [dict(x) for x in xsum["test"]]
        for sample_dp in test_set:
            article_lines = sample_dp["summary"].strip().split("\n") + sample_dp["document"].strip().split("\n")
            if do_truncate:
                article_lines = truncate(article_lines, tok, maxlen)
            yield {"input_lines": article_lines, "id":sample_dp["id"]}

    else:
        raise NotImplementedError
