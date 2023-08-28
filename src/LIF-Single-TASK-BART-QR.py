#!/usr/bin/env python
# coding: utf-8

# In[34]:


import transformers
transformers.__version__


# In[ ]:


# !pip install -U transformers==3.1.0


# In[ ]:


get_ipython().system('nvidia-smi')


# In[1]:


import torch
# from transformers import T5ForConditionalGeneration,T5Tokenizer
from transformers import BartForConditionalGeneration, BartTokenizer


def set_seed(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

set_seed(42)

model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')


# In[2]:


tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')


# In[3]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print ("device ",device)
model = model.to(device)


# In[ ]:


# import pickle
# train = pickle.load(open('qr-train.pkl', 'rb'))
# dev = pickle.load(open('qr-dev.pkl', 'rb'))
# test = pickle.load(open('qr-test.pkl', 'rb'))


# In[4]:


LEARNING_RATE = 1e-4
optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)


# In[5]:


from tqdm.notebook import tqdm


# In[6]:


import pandas as pd


# In[7]:


import pickle


# In[8]:


train = pd.read_pickle('data-lif-canard/train_canard_lif.pkl')


# In[ ]:


# special_tokens_dict = {'additional_special_tokens': ['<extra_id_0>','<extra_id_1>']}

# tokenizer.add_special_tokens(special_tokens_dict)


# In[9]:


train.head(1)


# In[10]:


tokenizer.add_tokens('<sep>')
model.resize_token_embeddings(len(tokenizer))
model.to(device)


# In[11]:


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
    
    input_s += ' <sep> '+ ' <sep> '.join(history)
    
    
    
#     if row['turn_number'].endswith('_1'):
#         input_s = 'paraphrase: ' + row['candidate']
        
#         if len(row['title'].strip())>0:
#             input_s += ' <extra_id_2> ' + row['title']
        
#         if len(row['description'].strip())>0:
#             input_s += ' <extra_id_2> ' + row['description']
            
#         input_seq.append(input_s)
#         labels_seq.append('shift '+ row['qr'])
#         continue
#     history = row['prev_qs'][::-1]
#     input_s = 'paraphrase: ' + row['candidate']
#     if len(history) > 0:
#         input_s += ' <extra_id_0> '+ ' <extra_id_1> '.join(history)
        
#     if len(row['title'].strip())>0:
#         input_s += ' <extra_id_2> ' + row['title']
        
#     if len(row['description'].strip())>0:
#         input_s += ' <extra_id_2> ' + row['description']
        
    input_seq.append(input_s)
    labels_seq.append(row['qr'])
#     labels_seq.append([row['qr'], row['label']])
#     if row['label']==0:
#         labels_seq.append('follow '+ row['qr'])
#     else:
#         labels_seq.append('shift '+ row['qr'])


# In[12]:


len(input_seq), len(labels_seq)


# In[ ]:


# input_seq = []
# labels_seq = []
# for data in train:
#     if len(data['history']) > 0:
#         history = [q['question'] for q in data['history']]
# #         input_seq.append(data['question']+ ' <extra_id_0> '+ ' <extra_id_1> '.join(history[::-1]))
#         input_seq.append(data['question']+ ' <s> '+ ' <s> '.join(history[::-1]))
#     else:
#         input_seq.append(data['question'])
#     labels_seq.append(data['output'])


# In[13]:


input_seq[0:3], labels_seq[0:3]


# In[ ]:


tokenizer.special_tokens_map


# In[14]:


tokenizer('<sep>')


# In[15]:


tokenizer.unk_token_id


# In[ ]:


tokenizer(input_seq[0], return_tensors="pt")


# In[16]:


for _ in range(5):
    for index, (input_s, label_s) in enumerate(tqdm(zip(input_seq, labels_seq), total=len(input_seq))):
        model.train()
        ip = tokenizer(input_s, return_tensors="pt").input_ids
        ip_att = tokenizer(input_s, return_tensors="pt").attention_mask
        lb = tokenizer(label_s, return_tensors="pt").input_ids
        ip=ip.to(device)
        lb=lb.to(device)
        ip_att=ip_att.to(device)

        loss = model(input_ids = ip, attention_mask=ip_att, labels=lb, return_dict=True).loss
        if index%1000 == 0:
            print(f'Epoch: {_}:{index}, Loss:  {loss.item()}')
        
#         if index% 24 ==0:
#         optimizer.zero_grad()
        loss.backward()
        optimizer.step()
#         
        optimizer.zero_grad()


# In[51]:


def qr(sentence):
    model.eval()
    max_len = 256
    
    text =  "paraphrase: " + sentence + ' </s>'
#     text = sentence ,pad_to_max_length=True
    encoding = tokenizer.encode_plus(text, return_tensors="pt")
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


# In[ ]:


# res = []
# for sent in input_seq:
#     res.append(qr(sent))
    


# In[ ]:


# len(res), len(labels_seq)


# In[ ]:


# for r,l in zip(res,labels_seq):
#     print(r)
#     print(l)
#     print('='*100)


# In[18]:


df_test= pd.read_pickle('data-lif-canard/topics_lif_with_t5_mtl.pkl')


# In[19]:


df_test.head(2)


# In[ ]:





# In[20]:


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
#     input_s = 'paraphrase: ' + row['query']
    
    history = []
    for ht in row['history']:
        history.append(ht)
    
    input_s += ' <sep> '+ ' <sep> '.join(history[::-1])
        
    input_seq_eval.append(input_s)


# In[21]:


len(input_seq_eval), len(qids)


# In[52]:


print(input_seq_eval[13])
print(qr(input_seq_eval[13]))


# In[22]:


input_seq_eval[0:5]


# In[23]:


res_eval = {}
for tnum, sent in tqdm(zip(qids,input_seq_eval), total=len(qids)):
#     if tnum.endswith('_1'):
#         res_eval[tnum] = sent.replace('<extra_id_0>','').strip()
#     else:
    res_eval[tnum] = qr(sent)


# In[ ]:


# for r,l in zip(res_eval,input_seq_eval):
#     print(l)
#     print(r)
#     print('='*100)


# In[24]:


res_eval_tmp = res_eval.copy()


# In[ ]:


for k,v in res_eval.items():
#     if k.endswith('_1'):
#         continue
#     else:
    res_eval[k] = ' '.join(v.split()[1:])


# In[25]:


input_seq_eval[0:5]


# In[26]:


res_eval


# In[ ]:


# cast19_eval = pd.read_pickle('cast19/castur_cast_eval_with_bart_qr_with_bart_mtl.pkl')


# In[ ]:


# cast19_eval.head(2)


# In[27]:


# df_test_t5_mtl= pd.read_pickle('data-lif-canard/test-retrieval.pkl')
df_test_t5_mtl = pd.read_pickle('data-lif-canard/topics_lif_with_t5_mtl_with_bart_mtl_with_t5basemtl.pkl')


# In[28]:


df_test_t5_mtl.head(2)


# In[29]:


df_test_t5_mtl['bart_stl'] = df_test_t5_mtl.apply(lambda row: res_eval[row['qid']] ,axis=1)


# In[30]:


# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 100)
pd.set_option("display.max_colwidth", 200)


# In[31]:


df_test_t5_mtl.sample(5)


# In[32]:


df_test_t5_mtl.to_pickle('data-lif-canard/topics_lif_with_t5_mtl_with_bart_mtl_with_t5basemtl_with_bartstl.pkl')


# In[ ]:





# In[ ]:





# ## Testing on LIF dataset

# In[ ]:


test_i = pd.read_pickle('data-lif-canard/lif_test_i.pkl')
test_ii = pd.read_pickle('data-lif-canard/lif_test_ii.pkl')
test_iii = pd.read_pickle('data-lif-canard/lif_test_iii.pkl')


# In[ ]:


from collections import Counter


# In[ ]:


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


# In[ ]:


labels = ['follow' if l==0 else 'shift' for l in labels_seq]


# In[ ]:


from sklearn.metrics import classification_report

print(classification_report(labels, pred_seq, digits=4))


# In[ ]:


Counter(pred_seq)


# In[ ]:


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


# In[ ]:


labels_ii = ['follow' if l==0 else 'shift' for l in labels_ii]


# In[ ]:


Counter(pred_ii)


# In[ ]:


print(classification_report(labels_ii, pred_ii, digits=4))


# In[ ]:





# In[ ]:


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


# In[ ]:


labels_iii = ['follow' if l==0 else 'shift' for l in labels_iii]


# In[ ]:


print(classification_report(labels_iii, pred_iii, digits=3))


# In[ ]:




