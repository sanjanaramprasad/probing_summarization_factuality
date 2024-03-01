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

    elif dsname=="reddit":
        import nltk
        reddit = load_dataset("reddit_tifu", "long")
        reddit = reddit.shuffle(seed=1729)
        train_set = [dict(x) for x in reddit["train"]]

        # manually selected ids that do not contain profanity
        chosen_ids = [3, 6, 8, 9, 12, 13, 14, 17, 18, 19,
                        21, 23, 31, 34, 37, 38, 39, 41, 48, 83,
                        50, 51, 54, 61, 72, 73, 76, 78, 79, 80]

        for idx in chosen_ids:
            sample_dp = train_set[idx]

            article_lines = sample_dp["documents"].strip().split("\n")
            temp = []
            for sent in article_lines:
                sent = sent.strip()
                if not sent:
                    continue
                temp.extend(nltk.sent_tokenize(sent))
            article_lines = temp

            if do_truncate:
                article_lines = truncate(article_lines, tok, maxlen)
            yield {"input_lines": article_lines, "id":idx}

    elif dsname=="samsum":
        samsum = load_dataset("samsum")
        test_set = [dict(x) for x in samsum["test"]]
        for sample_dp in test_set:
            article_lines = sample_dp["dialogue"].strip().split("\r\n")
            if (len(article_lines)<10):
                # ignore the conversations that are too short
                continue
            if do_truncate:
                article_lines = truncate(article_lines, tok, maxlen)
            yield {"input_lines": article_lines, "id":sample_dp["id"]}



    else:
        raise NotImplementedError
