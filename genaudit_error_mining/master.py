
import argparse
import pdb
import time

import jsonlines
import numpy as np
import torch
from torch.multiprocessing import Process, set_start_method

from factchecking_server_multiproc import process_root as process_root_factcheck
from generation_server_multiproc import process_root as process_root_generation
from datasets import load_dataset
from tqdm import tqdm
import requests
import json


from threading import Thread
from threading import Lock

from utils import get_data

def thread_root(thread_idx, num_threads, base_port, dsiter, outfilehandler, lock, gen_dosample, gen_temperature, gen_top_p):
    for (i,dp) in enumerate(tqdm(dsiter)):

        # if i==100:
        #     break

        if i%num_threads!=thread_idx:
            continue


        # print("THREAD ", thread_idx, "processing example id", i)

        GEN_ENDPOINT = f"http://localhost:{base_port+thread_idx}"
        VERIFY_ENDPOINT = f"http://localhost:{base_port+num_threads+thread_idx}"

        # the GT summary was actually the first line of article. so we prepend it to get full article
        full_doc = "\n".join(dp["input_lines"])

        if full_doc=="":
            print("ONE DOCUMENT COMPLETELY TRUNCATED OFF... SKIPPING THAT")
            continue

        input_dict = {"document":full_doc, "temperature":gen_temperature, "top_p":gen_top_p, "dosample": gen_dosample}
        r = requests.post(url=f"{GEN_ENDPOINT}/predict", json=input_dict)
        response = r.text
        output_obj = json.loads(response)
        if not output_obj["success"]:
            continue

        input_dict["summary"] = output_obj["result"]
        # print("YYYYYYYYYYYYYYYYYY", input_dict)
        r = requests.post(url=f"{VERIFY_ENDPOINT}/predict", json=input_dict)
        response = r.text
        output_obj = json.loads(response)
        if not output_obj["success"]:     # this almost always happens due to OOM errors
            continue

        result = output_obj["result"]

        factcheck_preds = result["factcheck_predictions"]
        if any(len(x["todelete_spans"])>0 for x in factcheck_preds):
            print("ERRORS found!")
        else:
            # COMMENTING THIS OUT SINCE WE WANT TO LOG ALL CASES EVEN IF NO ERROR FOUND
            # continue
            pass

        result["id"] = dp["id"]

        with lock:
            out_str = json.dumps(result, ensure_ascii=False)
            outfilehandler.write(out_str+"\n")
            if thread_idx==0:
                outfilehandler.flush()


if __name__=="__main__":

    np.random.seed(1337)

    parser = argparse.ArgumentParser()
    parser.add_argument("--gen-model-name", type=str, required=True)
    parser.add_argument("--fact-model-name", type=str, required=True)
    parser.add_argument("--gen-quantize", type=str, default="16bit")
    parser.add_argument("--base-port", type=int, default=9000)
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--max-input-len", type=int, default=1000)
    parser.add_argument("--max-decode-len", type=int, default=150)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--gen-dosample", action="store_true")
    parser.add_argument("--gen-temperature", type=float, default=1.0)
    parser.add_argument("--gen-top-p", type=float, default=None)
    parser.add_argument("--dsname", type=str, required=True)
    parser.add_argument("--do-truncate", action="store_true")
    parser.add_argument("--max-predict", type=int, default=99999999)


    args = parser.parse_args()

    for arg in vars(args):
        print(arg, getattr(args, arg))

    gen_model_name = args.gen_model_name
    output_file = args.output_file
    fact_model_name = args.fact_model_name
    base_port = args.base_port
    gen_quantize = args.gen_quantize
    max_input_len = args.max_input_len
    max_decode_len = args.max_decode_len
    num_processes = args.num_processes
    gen_dosample = args.gen_dosample
    gen_temperature = args.gen_temperature
    gen_top_p = args.gen_top_p
    dsname = args.dsname
    do_truncate = args.do_truncate
    max_predict = args.max_predict

    set_start_method("spawn")


    generation_procs = []
    for pidx in range(num_processes):
        p = Process(target=process_root_generation, args=(pidx, pidx, gen_model_name, base_port, gen_quantize, 9999999, max_decode_len))
        p.start()
        generation_procs.append(p)

    factcheck_procs = []
    for pidx in range(num_processes):
        p = Process(target=process_root_factcheck, args=(pidx, pidx+num_processes, fact_model_name, base_port+num_processes, 99999999, 9999999))
        p.start()
        factcheck_procs.append(p)

    # pdb.set_trace()

    for port in range(base_port, base_port+2*num_processes):
        while True:
            try:
                r = requests.get(url=f"http://localhost:{port}/test")
                response = r.text
                print(response)
                print(f"Contact established with port {port}! proceeding...")
                break
            except requests.exceptions.ConnectionError:
                print("Waiting for backend to start. Will ping again in 5 seconds...")
                time.sleep(5)

    ds = get_data(dsname, do_truncate=do_truncate, maxlen=max_input_len)
    ds = list(ds)  # this was an iterator and was to be passed to multiple threads. that would have created problems and perhaps each dp wouldve gone to just one thread
    np.random.seed(1729)
    np.random.shuffle(ds)
    # pdb.set_trace()
    ds = ds[:max_predict]

    master_lock = Lock()
    writer = open(output_file, "w")

    threads = [Thread(target=thread_root,
                      args=(thread_idx, num_processes, base_port, ds, writer, master_lock, gen_dosample, gen_temperature, gen_top_p))
                           for thread_idx in range(num_processes)]
    # start threads
    for thread in threads:
        thread.start()
    # wait for threads to finish
    for thread in threads:
        thread.join()

    writer.close()

    [p.kill() for p in generation_procs]
    [p.kill() for p in factcheck_procs]
