

The script below gives takes a summarization dataset, generates summaries using mistral, then uses evidence-inspector model to generate factchecking reports (evidence+suggested edits), then saves it on disk.
Most arguments are self-explanatory, except one - the num-processes  argument tells you how many parallel jobs to run. One job needs to spawn two models (one to generate summary another to factcheck it. So you need to have 2*num_processes number of gpus (or at least that's how it's code right now - every model on separate gpu).
  
`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python master.py  
--gen-model-name mistralai/Mistral-7B-Instruct-v0.1  
--fact-model-name kundank/evinspect-usb-flanul2-qlora4bit  
--output-file ./xsum-mistral-topp_0.9-temp_1.0.jsonl  
--base-port 8900 --num-processes 4 --gen-dosample  
--gen-top-p 0.9 --gen-temperature 1.0 --dsname xsum  
--do-truncate --max-input-len 2000 --max-predict 40`
