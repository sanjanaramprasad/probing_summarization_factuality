import json
import openai
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

with open('/home/ramprasad.sa/secrets/byron_openai.json', 'r') as fp:
    secrets = json.load(fp)

    
OPENAI_API_KEY = secrets['openapi_key']
openai.api_key = OPENAI_API_KEY



model_path = {'mistral7b': 'mistralai/Mistral-7B-Instruct-v0.1',
             'falcon7b': 'tiiuae/falcon-7b-instruct',
             'flanul2': 'google/flan-ul2'}





def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_path[model_name],
                                         cache_dir = '/scratch/ramprasad.sa/huggingface_models')

    model = AutoModelForCausalLM.from_pretrained(model_path[model_name],
                                            cache_dir = '/scratch/ramprasad.sa/huggingface_models')
    model = model.to('cuda')
    return tokenizer, model



def read_jsonl(filepath):
    data = []
    with open(filepath) as f:
        for line in f:
            data.append(json.loads(line))
    return data




# @retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(2))
def get_chatgpt_response(prompt, model = "gpt-4-0613"):
        response = openai.ChatCompletion.create(model=model,
                                       messages=[
                                           
                        {"role": "user", "content": f'{prompt}'},   
                        ], 
                        )
        return response['choices'][0]['message']['content']




def scrub(text):
    words = text.split(' ')
    words = [re.sub('[^a-zA-Z0-9]+\s*', ' ', w).strip().lower() for w in words]
    words = [w for w in words if w.strip()]
    text = ' '.join(words)
    return text
    
def postprocess_spans(all_spans):
    assert(type(all_spans) is list)
    all_spans = [each for each in all_spans if each and each.lower() != 'none']
    all_spans = [scrub(each) for each in all_spans ]
    return all_spans




def text_inconsistent(response):
    inconsistent_phrases = ['no inconsisten', 'none', '[]', 'is consistent']
    if not response.strip():
        return True
    
    elif [each  for each in inconsistent_phrases if each in response.lower()]:
        return True
    return False


def retrieve_ann_spans(nonfactual_spans,
                       prediction_type = 'annotate',
                      ):

    if prediction_type == 'annotate' or '<span>' in nonfactual_spans:
            all_spans = re.findall("<span>(.*?)</span>", nonfactual_spans, re.DOTALL)
              
    else:
            list_spans = re.findall("\[(.*?)\]", nonfactual_spans, re.DOTALL)
            if not list_spans:
                all_spans = nonfactual_spans.split(',')
            else:
                try:
                    all_spans = eval(nonfactual_spans) 
                except:
                    print('EXCEPTION', nonfactual_spans, list_spans)
                    all_spans = list_spans[0].split(',')
    return all_spans
    

def extract_nonfact_spans(nonfactual_spans, 
                              prediction_type = 'predict', 
                              evaluation = False,
                                compare = None):

    
    ### check if the span is a list ###
    if type(nonfactual_spans) is list:
        all_spans = [each for each in nonfactual_spans if each]
        
    elif text_inconsistent(nonfactual_spans):
        all_spans = []
    
    else:
        all_spans = retrieve_ann_spans(nonfactual_spans,
                                      prediction_type = prediction_type)

    if evaluation:
        all_spans = postprocess_spans(all_spans)
    else:
        all_spans = [each.strip('"') for each in all_spans]
        all_spans = [each for each in all_spans if each and each.lower() != 'none']
    
    if compare:
        all_spans = [each for each in all_spans if get_fuzzy_score(each, compare) > 0.5]
        # for each in all_spans:
        #     print(compare, each, get_fuzzy_score(each, compare))
    return all_spans



def longest_common_substring_ignore_punctuations(str1, str2):
    # Remove punctuations from strings
    # translator = str.maketrans('', '', string.punctuation)
    # str1 = str1.translate(translator)
    # str2 = str2.translate(translator)

    len1, len2 = len(str1), len(str2)
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    max_len, end_index = 0, 0

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > max_len:
                    max_len = dp[i][j]
                    end_index = i - 1
            else:
                dp[i][j] = 0

    start_index = end_index - max_len + 1
    longest_substring = str1[start_index:end_index + 1]
    
    return longest_substring

