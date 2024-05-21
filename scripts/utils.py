import json
import openai
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModelForCausalLM

with open('/home/ramprasad.sa/secrets/byron_openai.json', 'r') as fp:
    secrets = json.load(fp)

    
OPENAI_API_KEY = secrets['openapi_key']
openai.api_key = OPENAI_API_KEY



model_path = {'mistral7b': 'mistralai/Mistral-7B-Instruct-v0.1',
             'falcon7b': 'tiiuae/falcon-7b-instruct',
             'llama7b': '/work/frink/models/llama_2_7b_chat_hf',
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
                          summary,
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
    all_spans = [longest_common_substring_ignore_punctuations(span, summary) for span in all_spans]
    all_spans = [each.strip() for each in all_spans if each.strip()]
    return all_spans




def get_l1(probs_with_source, probs_without_source):
    all_diff = []
    for i in range(0, len(probs_with_source)):
        all_diff += [abs(probs_with_source[i] - probs_without_source[i])]
    return sum(all_diff)



def get_document_keywords(document, 
                         num_keywords = 5):
    # Tokenization
    tokens = word_tokenize(document)

    # Lowercasing and removing stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word.lower() for word in tokens if word.isalnum() and word.lower() not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

    # Join the tokens back into a string
    processed_document = ' '.join(lemmatized_tokens)

    # Calculate TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([processed_document])

    # Get feature names (words)
    feature_names = tfidf_vectorizer.get_feature_names_out()

    # Get TF-IDF scores for each word
    tfidf_scores = tfidf_matrix.toarray()[0]

    # Create a dictionary with words and their TF-IDF scores
    word_scores = dict(zip(feature_names, tfidf_scores))

    # Sort the words based on TF-IDF scores
    sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)

    # Extract top keywords
    
    keywords = [word for word, score in sorted_words[:num_keywords]]

    #print("Keywords:", keywords)
    return keywords
