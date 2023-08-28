#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# In[2]:


import nltk
nltk.download('punkt')


# In[3]:


model_checkpoint = "facebook/bart-base"


# In[4]:


import pandas as pd
pd.set_option('display.max_column',None)
pd.set_option('display.max_rows',None)
pd.set_option('display.max_seq_items',None)
pd.set_option('display.max_colwidth', 500)
pd.set_option('expand_frame_repr', True)


# In[5]:


# !pip install pickle5


# In[6]:


import pickle5 as pickle


# In[7]:


train = pickle.load(open('data-lif-canard/train_canard_lif.pkl','rb'))


# In[8]:


dev = pickle.load(open('data-lif-canard/dev_canard_lif.pkl','rb'))


# In[9]:


train.head(5)


# In[10]:


def convert_input(row):
    prev_ans = row['prev_ans'][::-1]
    prev_qa = row['prev_qa'][::-1]
    input_s = 'paraphrase: ' + row['candidate']
    history = []
    for q, a in zip(prev_qa,prev_ans):
        history.append(q)
        if a == 'CANNOTANSWER':
            continue
        history.append(a)
        
    
    input_s += ' <sep> '+ ' <sep> '.join(history)
    
#     input_s +=  ' <extra_id_0> ' +  row['candidate']
    
    
    return input_s


# In[11]:


def convert_output(row):
    if row['label']==1:
        labels_seq = 'follow '+ row['qr']
    else:
        labels_seq = 'shift '+ row['qr']
    return labels_seq


# In[12]:


train['text'] = train.apply(convert_input, axis=1)
train['output'] = train.apply(convert_output, axis=1)


# In[13]:


dev['text'] = dev.apply(convert_input, axis=1)
dev['output'] = dev.apply(convert_output, axis=1)


# In[14]:


train = train.rename({'label':'lbs'}, axis=1)
dev = dev.rename({'label':'lbs'}, axis=1)


# In[15]:


train.sample(1)


# In[16]:


from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer,AutoTokenizer


# In[17]:


tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


# In[18]:


# !pip install datasets


# In[19]:


from datasets.dataset_dict import DatasetDict
from datasets import Dataset


# In[20]:


raw_datasets = DatasetDict()
raw_datasets['train'] = Dataset.from_pandas(train)
raw_datasets['dev'] = Dataset.from_pandas(dev)
raw_datasets['test'] = Dataset.from_pandas(dev)


# In[21]:


max_input_length = 512
max_target_length = 256

def preprocess_function(examples):
    inputs = [doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs, padding=True, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["output"], padding=True, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# In[22]:


tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)


# In[23]:


model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)


# In[24]:


batch_size = 20
model_name = model_checkpoint.split("/")[-1]
output_path = f"{model_name}-lif-mtl-follow-qr"

args = Seq2SeqTrainingArguments(
    output_path,
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=10,
    num_train_epochs=5,
    predict_with_generate=True,
    fp16=False,
    push_to_hub=False,
    save_strategy ="epoch",
    logging_dir= output_path + '/logs',
    logging_strategy="epoch",
#     logging_steps=1000,
    load_best_model_at_end=True
#     save_steps=1000,
)


# In[25]:


# !pip install rouge_score


# In[26]:


data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
from datasets import load_metric
metric = load_metric("rouge")


# In[27]:


def split_tasks(text):
    if text.startswith('follow ') or text.startswith('shift '):
        return text.split()[0], ' '.join(text.split()[1:])
    else:
        return 'follow', text


# In[28]:


import numpy as np


# In[29]:


from sklearn.metrics import accuracy_score, f1_score
def compute_metrics_multitask(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    decoded_preds = [split_tasks(pred) for pred in decoded_preds]
    decoded_labels = [split_tasks(label) for label in decoded_labels]
    
    rel_preds = [m[0] for m in decoded_preds]
    rel_labels = [m[0] for m in decoded_labels]
    
    ans_preds = [m[1] for m in decoded_preds]
    ans_labels = [m[1] for m in decoded_labels]
    
    
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions[7:]]
    
    result = metric.compute(predictions=ans_preds, references=ans_labels, use_stemmer=True)
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    
    rel_acc = accuracy_score(rel_labels, rel_preds)
    
    result['Acc'] = rel_acc * 100
    
    return {k: round(v, 2) for k, v in result.items()}


# In[30]:


trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["dev"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics_multitask
)


# In[31]:


trainer.train()


# In[32]:


from transformers import set_seed


# In[ ]:





# In[33]:


def qr(sentence, model):
    model.eval()
    max_len = 256
    
    text =  "paraphrase: " + sentence
    encoding = tokenizer.encode_plus(text,pad_to_max_length=True, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

    set_seed(42)
    # set top_k = 50 and set top_p = 0.95 and num_return_sequences = 3
    beam_outputs = model.generate(
        input_ids=input_ids,# attention_mask=attention_masks,
        do_sample=True,
        max_length=256,
        top_k=120,
        top_p=0.98,
        early_stopping=True,
        num_return_sequences=1
    )


    final_outputs =[]
    for beam_output in beam_outputs:
        sent = tokenizer.decode(beam_output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
        if sent.lower() != sentence.lower() and sent not in final_outputs:
            final_outputs.append(sent)


    return final_outputs[0]


# In[34]:


df_test= pickle.load(open('data-lif-canard/test-retrieval.pkl','rb'))


# In[35]:


df_test.head()


# In[44]:


input_seq_eval = []
qids = []
# labels_seq_eval = []
for index, row in df_test.iterrows():
#     if row['turn_number'].endswith('_1'):
#         continue
    qids.append(row['qid'])
#     history = row['prev_qs'][::-1]
#     input_seq_eval.append(row['candidate']+ ' <extra_id_0> '+ ' <extra_id_1> '.join(history))
    input_s = row['query']  
    
    history = []
    for ht in row['history'][::-1]:
        history.append(ht)
    
    input_s += ' sep ' + ' <sep> '.join(history)
#     input_s +=  ' <extra_id_0> ' +  
    input_seq_eval.append(input_s)


# In[45]:


get_ipython().system('ls bart-base-lif-mtl-follow-qr/checkpoint-3142')


# In[46]:


import torch


# In[51]:


input_seq_eval[4]


# In[47]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print ("device ",device)


# In[48]:


from tqdm.notebook import tqdm


# In[49]:


from nltk.translate.bleu_score import sentence_bleu, corpus_bleu


# In[50]:


def eval_model(pred, lbls):
# preds = [s.split() for s in df_test['mtl-t5-qr'].to_list()]
# lbls = [s.split() for s in df_test['qr'].to_list()]
    all_bleu = []
    for p, l in zip(preds, lbls):
        bleu_score = sentence_bleu([l], p)
        all_bleu.append(round(bleu_score,3))
    return all_bleu


# In[55]:


for chk in os.listdir('bart-base-lif-mtl-follow-qr/'):
    if not chk.startswith('checkpoint'):
        continue
    print(chk)
    model.load_state_dict(torch.load('bart-base-lif-mtl-follow-qr/'+ chk + '/pytorch_model.bin'))
    model.to(device)
    model.eval()
    preds = []
    lbls = [s.split() for s in df_test['qr'].to_list()]
    for qid, sent in tqdm(zip(qids, input_seq_eval), total=len(qids)):
        res = qr(sent,model)
        res = ' '.join(res.split()[1:])
        preds.append(res)
        
    preds = [s.split() for s in preds]
    eval_ = eval_model(preds, lbls)
    print(sum(eval_)/len(eval_))


# In[56]:


get_ipython().system('ls bart-base-lif-mtl-follow-qr/')


# In[57]:


model.load_state_dict(torch.load('bart-base-lif-mtl-follow-qr/checkpoint-15710/pytorch_model.bin'))
model.to(device)
model.eval()
# preds = []
qid2qr = {}
for qid, sent in tqdm(zip(qids, input_seq_eval), total=len(qids)):

#     preds.append(qr(sent,model))
    qid2qr[qid] = qr(sent,model)


# In[58]:


# df_test['mtl-t5-qr'] = df_test['qid'].apply(lambda x:' '.join(qid2qr[x].split()[1:]))
df_test['mtl-t5-qr'] = df_test['qid'].apply(lambda x:qid2qr[x])


# In[ ]:


# sum(all_bleu)/len(all_bleu)


# In[62]:


df_test.head(50)


# In[ ]:




