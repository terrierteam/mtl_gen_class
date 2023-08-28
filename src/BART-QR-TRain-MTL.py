#!/usr/bin/env python
# coding: utf-8

# In[16]:


import transformers
transformers.__version__


# In[90]:


get_ipython().system('nvidia-smi')


# In[2]:


from transformers import BartTokenizer

tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')


# In[3]:


import torch

class PyTorchDatasetCreate(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels['input_ids'][idx])
        return item

    def __len__(self):
        return len(self.labels['input_ids'])


# In[4]:


import pandas as pd
from tqdm.notebook import tqdm


# In[5]:


train = pd.read_pickle('data-lif-canard/train_canard_lif.pkl')


# In[6]:


train.head(1)


# In[7]:


dev = pd.read_pickle('data-lif-canard/dev_canard_lif.pkl')


# In[8]:


dev.head(1)


# In[10]:


dev_input_seq = []
dev_labels_seq = []
for index, row in dev.iterrows():
    prev_ans = row['prev_ans'][::-1]
    prev_qa = row['prev_qa'][::-1]
    input_s = 'paraphrase: ' + row['candidate']
    
    history = []
    for q, a in zip(prev_qa,prev_ans):
        history.append(a)
        history.append(q)
    
    input_s += ' </s> '+ ' </s> '.join(history)
    
    dev_input_seq.append(input_s)
#     labels_seq.append(row['qr'])
#     labels_seq.append([row['qr'], row['label']])
    if row['label']==1:
        dev_labels_seq.append('follow '+ row['qr'])
    else:
        dev_labels_seq.append('shift '+ row['qr'])


# In[9]:


input_seq = []
labels_seq = []
for index, row in train.iterrows():
    prev_ans = row['prev_ans'][::-1]
    prev_qa = row['prev_qa'][::-1]
    input_s = 'paraphrase: ' + row['candidate']
    
    history = []
    for q, a in zip(prev_qa,prev_ans):
        history.append(a)
        history.append(q)
    
    input_s += ' </s> '+ ' </s> '.join(history)
    
    input_seq.append(input_s)
#     labels_seq.append(row['qr'])
#     labels_seq.append([row['qr'], row['label']])
    if row['label']==1:
        labels_seq.append('follow '+ row['qr'])
    else:
        labels_seq.append('shift '+ row['qr'])


# In[11]:


input_seq[0:2], labels_seq[0:2]


# In[12]:


# tokenizer.add_tokens('<sep>')
# tokenizer('<sep>')


# In[12]:


train_encodings = tokenizer(input_seq, truncation=True, padding=True)
train_label_encodings = tokenizer(labels_seq, truncation=True, padding=True)


# In[13]:


val_encodings = tokenizer(dev_input_seq, truncation=True, padding=True)
val_label_encodings = tokenizer(dev_labels_seq, truncation=True, padding=True)


# In[14]:


train_dataset = PyTorchDatasetCreate(train_encodings, train_label_encodings)
val_dataset = PyTorchDatasetCreate(val_encodings, val_label_encodings)


# In[17]:


from transformers import BartForConditionalGeneration, Trainer, TrainingArguments

training_args = TrainingArguments(
output_dir='./bart-results-mtl-2', # output directory
num_train_epochs=5, # total number of training epochs
per_device_train_batch_size=1, # batch size per device during training
per_device_eval_batch_size=1, # batch size for evaluation
warmup_steps=200, # number of warmup steps for learning rate scheduler
weight_decay=0.01, # strength of weight decay
logging_dir='./bart-logs-mtl-2', # directory for storing logs
logging_steps=10000, save_steps=62839, do_eval=True)

model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
# model.resize_token_embeddings(len(tokenizer))

trainer = Trainer(
model=model, # the instantiated :hugs: Transformers model to be trained
args=training_args, # training arguments, defined above
train_dataset=train_dataset, # training dataset
eval_dataset=val_dataset) # evaluation dataset


# In[18]:


trainer.train()


# In[19]:


model.save_pretrained('bart-model/bart-mtl-4.1-with-sep')


# In[20]:


input_seq[0]


# In[21]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print ("device ",device)


# In[22]:


def qr(sentence):
    model.eval()
    max_len = 256
    
    text =  "paraphrase: " + sentence + ' </s>'
#     text = sentence
    encoding = tokenizer.encode_plus(text, max_length=512, return_tensors="pt", truncation=True)
    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)


    # set top_k = 50 and set top_p = 0.95 and num_return_sequences = 3
    beam_outputs = model.generate(
        input_ids=input_ids,# attention_mask=attention_masks,
        do_sample=True,
        max_length=256,
        top_k=120,
        top_p=0.98,
        early_stopping=True,
        num_return_sequences=20
    )


#     print ("\nOriginal Question ::")
#     print (input_seq[ids])
#     print ("\n")
#     print ("Paraphrased Questions :: ")
    final_outputs =[]
    for beam_output in beam_outputs:
        sent = tokenizer.decode(beam_output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
        if sent.lower() != sentence.lower() and sent not in final_outputs:
            final_outputs.append(sent)

#     for i, final_output in enumerate(final_outputs):
#         print("{}: {}".format(i, final_output))
    return final_outputs[0]


# In[22]:


# model.save_pretrained('bart-model/bart-mtl-4.1')


# In[23]:


print(input_seq[4])
qr(input_seq[4])


# In[24]:


df_test= pd.read_pickle('data-lif-canard/topics_lif_with_t5_mtl.pkl')


# In[25]:


df_test.head(2)


# In[73]:


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
    for ht in row['history']:
        history.append(ht)
    
    input_s += ' </s> '+ ' </s> '.join(history[::-1])
        
    input_seq_eval.append(input_s)


# In[74]:


print(input_seq_eval[3])
print(qr(input_seq_eval[3]))


# In[75]:


print(input_seq_eval[1])
print(qr(input_seq_eval[1]))


# In[76]:


res_eval = {}
for tnum, sent in tqdm(zip(qids,input_seq_eval), total=len(qids)):
#     if tnum.endswith('_1'):
#         res_eval[tnum] = sent.replace('<extra_id_0>','').strip()
#     else:
    res_eval[tnum] = qr(sent)


# In[77]:


res_eval_tmp = res_eval.copy()


# In[78]:


for k,v in res_eval.items():
#     if k.endswith('_1'):
#         continue
#     else:
    res_eval[k] = ' '.join(v.split()[1:])


# In[79]:


for i, (k,v) in enumerate(res_eval.items()):
    if i < 10:
        print(k, v)
    else:
        break


# In[80]:


df_test_t5_mtl= pd.read_pickle('data-lif-canard/topics_lif_with_t5mtl-bartmtl.pkl')


# In[81]:


df_test_t5_mtl.head(2)


# In[82]:


df_test_t5_mtl['bartmtl2_qr'] = df_test_t5_mtl.apply(lambda row: res_eval[row['qid']] ,axis=1)


# In[83]:


# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 100)
pd.set_option("display.max_colwidth", 200)


# In[84]:


df_test_t5_mtl.sample(5)


# In[85]:


df_test_t5_mtl.to_pickle('data-lif-canard/topics_lif_with_t5mtl-bartmtl.pkl')


# ## Testing on LIF dataset

# In[53]:


test_i = pd.read_pickle('data-lif-canard/lif_test_i.pkl')
test_ii = pd.read_pickle('data-lif-canard/lif_test_ii.pkl')
test_iii = pd.read_pickle('data-lif-canard/lif_test_iii.pkl')


# In[54]:


from collections import Counter


# In[61]:


pred_seq = []
labels_seq = []
for index, row in tqdm(test_i.iterrows(), total=len(test_i)):
    
#     input_s = 'paraphrase: ' + row['query']
    input_s = row['query']
    history = row['history'][::-1]
    
    
    input_s += ' </s> '+ ' </s> '.join(history)
    
    res = qr(input_s)
    
    pred_seq.append(res.split()[0])
    
    labels_seq.append(row['label'])


# In[62]:


labels = ['follow' if l==1 else 'shift' for l in labels_seq]


# In[63]:


from sklearn.metrics import classification_report

print(classification_report(labels, pred_seq, digits=3))


# In[64]:


Counter(pred_seq)


# In[87]:


import pickle
# !mkdir lif-results
pickle.dump(pred_seq, open('lif-results/mtl-bart-4.1-lif-test-i.pkl','wb'))


# In[65]:


pred_ii = []
labels_ii = []
for index, row in tqdm(test_ii.iterrows(), total=len(test_ii)):
    
#     input_s = 'paraphrase: ' + row['query']
    input_s = row['query']
    history = row['history'][::-1]
    
    
    input_s += ' </s> '+ ' </s> '.join(history)
    
    res = qr(input_s)
    
    pred_ii.append(res.split()[0])
    
    labels_ii.append(row['label'])


# In[66]:


labels_ii = []
for index, row in tqdm(test_ii.iterrows(), total=len(test_ii)):
    labels_ii.append(row['label'])


# In[67]:


labels_ii = ['follow' if l==1 else 'shift' for l in labels_ii]


# In[68]:


Counter(labels_ii)


# In[69]:


Counter(pred_ii)


# In[70]:


print(classification_report(labels_ii, pred_ii, digits=3))


# In[88]:


pickle.dump(pred_ii, open('lif-results/mtl-bart-4.1-lif-test-ii.pkl','wb'))


# In[58]:


pred_iii = []
labels_iii = []
for index, row in tqdm(test_iii.iterrows(), total=len(test_iii)):
    
#     input_s = 'paraphrase: ' + row['query']
    input_s = row['query']
    history = row['history'][::-1]
    
    
    input_s += ' </s> '+ ' </s> '.join(history)
    
    res = qr(input_s)
    
    pred_iii.append(res.split()[0])
    
    labels_iii.append(row['label'])


# In[59]:


labels_iii = ['follow' if l==1 else 'shift' for l in labels_iii]


# In[60]:


print(classification_report(labels_iii, pred_iii, digits=3))


# In[36]:


# model.save_pretrained('bart-model/bart-mtl')


# In[89]:


pickle.dump(pred_iii, open('lif-results/mtl-bart-4.1-lif-test-iii.pkl','wb'))


# In[ ]:




