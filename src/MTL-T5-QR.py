#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import transformers
transformers.__version__


# In[ ]:


# !pip install -U transformers==3.1.0


# In[ ]:


get_ipython().system('nvidia-smi')


# In[ ]:


import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer
# from transformers import BartForConditionalGeneration, BartTokenizer


def set_seed(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

set_seed(42)

model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_paraphraser')


# In[ ]:


tokenizer = T5Tokenizer.from_pretrained('t5-base')


# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print ("device ",device)
model = model.to(device)


# In[ ]:


# import pickle
# train = pickle.load(open('qr-train.pkl', 'rb'))
# dev = pickle.load(open('qr-dev.pkl', 'rb'))
# test = pickle.load(open('qr-test.pkl', 'rb'))


# In[ ]:


LEARNING_RATE = 1e-4
optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)


# In[ ]:


from tqdm.notebook import tqdm


# In[ ]:


import pandas as pd


# In[ ]:


castur_cast19 = pd.read_pickle('castur_cast_train.pkl')


# In[ ]:


# special_tokens_dict = {'additional_special_tokens': ['<extra_id_0>','<extra_id_1>']}

# tokenizer.add_special_tokens(special_tokens_dict)


# In[ ]:


castur_cast19.head(1)


# In[ ]:


cast19_title_dest = pd.read_pickle('cast19/trec_cast19_eval_org.pkl')
cast19_title_dest=cast19_title_dest[['turn_number', 'title', 'description','qr', 'candidate']]
cast19_title_dest.head()


# In[ ]:


castur_cast19.head(2)


# In[ ]:


castur_cast19 = castur_cast19.merge(cast19_title_dest, how='right')


# In[ ]:


castur_cast19.head()


# In[ ]:


input_seq = []
labels_seq = []
for index, row in castur_cast19.iterrows():
    if row['turn_number'].endswith('_1'):
        input_s = 'paraphrase: ' + row['candidate']
        
        if len(row['title'].strip())>0:
            input_s += ' <extra_id_2> ' + row['title']
        
        if len(row['description'].strip())>0:
            input_s += ' <extra_id_2> ' + row['description']
            
        input_seq.append(input_s)
        labels_seq.append('shift '+ row['qr'])
        continue
    history = row['prev_qs'][::-1]
    input_s = 'paraphrase: ' + row['candidate']
    if len(history) > 0:
        input_s += ' <extra_id_0> '+ ' <extra_id_1> '.join(history)
        
    if len(row['title'].strip())>0:
        input_s += ' <extra_id_2> ' + row['title']
        
    if len(row['description'].strip())>0:
        input_s += ' <extra_id_2> ' + row['description']
        
    input_seq.append(input_s)
#     labels_seq.append([row['qr'], row['label']])
    if row['label']==0:
        labels_seq.append('follow '+ row['qr'])
    else:
        labels_seq.append('shift '+ row['qr'])


# In[ ]:


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


# In[ ]:


input_seq[0:3], labels_seq[0:3]


# In[ ]:


tokenizer(input_seq[0], return_tensors="pt")


# In[ ]:


# lbs = [l[0] for l in labels_seq]
# lbs_tps = [l[1] for l in labels_seq]
# len(lbs_tps), len(lbs)


# In[ ]:


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
        if index%100 == 0:
            print(f'Epoch: {_}:{index}, Loss:  {loss.item()}')
        
#         if index% 24 ==0:
#         optimizer.zero_grad()
        loss.backward()
        optimizer.step()
#         
        optimizer.zero_grad()


# In[ ]:


def qr(sentence):
    model.eval()
    max_len = 256
    
    text =  "paraphrase: " + sentence
    encoding = tokenizer.encode_plus(text,pad_to_max_length=True, return_tensors="pt")
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


res = []
for sent in input_seq:
    res.append(qr(sent))
    


# In[ ]:


len(res), len(labels_seq)


# In[ ]:


for r,l in zip(res,labels_seq):
    print(r)
    print(l)
    print('='*100)


# In[ ]:





# In[ ]:


castur_cast19_eval = pd.read_pickle('cast19/castur_cast_eval.pkl')
castur_cast19_eval.head(2)


# In[ ]:


train_cast19_org = pd.read_pickle('cast19/trec_cast19_train_org.pkl')


# In[ ]:


train_cast19_org.head()


# In[ ]:





# In[ ]:


input_seq_eval = []
turn_number = []
# labels_seq_eval = []
for index, row in train_cast19_org.iterrows():
#     if row['turn_number'].endswith('_1'):
#         continue
    turn_number.append(row['turn_number'])
#     history = row['prev_qs'][::-1]
#     input_seq_eval.append(row['candidate']+ ' <extra_id_0> '+ ' <extra_id_1> '.join(history))
    if row['turn_number'].endswith('_1'):
        input_s = 'paraphrase: ' + row['candidate']
        
        if len(row['title'].strip())>0:
            input_s += ' <extra_id_2> ' + row['title']
        
        if len(row['description'].strip())>0:
            input_s += ' <extra_id_2> ' + row['description']
            
        input_seq_eval.append(input_s)
        
        continue
    history = row['prev_qs'][::-1]
    input_s = 'paraphrase: ' + row['candidate']
    if len(history) > 0:
        input_s += ' <extra_id_0> '+ ' <extra_id_1> '.join(history)
        
    if len(row['title'].strip())>0:
        input_s += ' <extra_id_2> ' + row['title']
        
    if len(row['description'].strip())>0:
        input_s += ' <extra_id_2> ' + row['description']
        
    input_seq_eval.append(input_s)


# In[ ]:


len(input_seq_eval)


# In[ ]:


input_seq_eval[0:2]


# In[ ]:


res_eval = {}
for tnum, sent in tqdm(zip(turn_number,input_seq_eval), total=len(turn_number)):
#     if tnum.endswith('_1'):
#         res_eval[tnum] = sent.replace('<extra_id_0>','').strip()
#     else:
    res_eval[tnum] = qr(sent)


# In[ ]:


for r,l in zip(res_eval,input_seq_eval):
    print(l)
    print(r)
    print('='*100)


# In[ ]:


res_eval


# In[ ]:


res_eval


# In[ ]:


for k,v in res_eval.items():
#     if k.endswith('_1'):
#         continue
#     else:
    res_eval[k] = ' '.join(v.split()[1:])


# In[ ]:


cast19_eval = pd.read_pickle('cast19/castur_cast_eval_with_bart_qr_with_bart_mtl.pkl')


# In[ ]:


cast19_eval.head(2)


# In[ ]:


cast19_eval['t5_mtl_qr'] = cast19_eval.apply(lambda row: res_eval[row['turn_number']] ,axis=1)


# In[ ]:


# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 100)
pd.set_option("display.max_colwidth", 200)


# In[ ]:


cast19_eval.head(2)


# In[ ]:


cast19_eval.to_pickle('cast19/castur_cast_eval_with_bart_qr_with_bart_mtl.pkl')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


get_ipython().system('mkdir bart-model')


# In[ ]:


torch.save(model, 'bart-model/t5-model-cannard')


# ## CAsT 2019 data

# In[ ]:


import pandas as pd


# In[ ]:


cast19=pd.read_pickle('cast19-qr.pkl')


# In[ ]:


input_seq = []
labels_seq = []
for index, data in cast19.iterrows():
    
    query = ''
    if len(data['history']) > 0:
#         history = [q['raw_utterance'] for q in data['history']]
        query = data['raw_utterance']+ ' <s> '+ ' <s> '.join(history[::-1])
        if len(data['description'].strip()) > 0:
            query += ' <s> ' + data['description'].strip()
    else:
        query = data['raw_utterance']

    input_seq.append(query)
    labels_seq.append(data['qr'])


# In[ ]:


input_seq[4],labels_seq[4]


# In[ ]:


for _ in range(5):
    for index, (input_s, label_s) in enumerate(tqdm(zip(input_seq, labels_seq), total=len(labels_seq))):
        model.train()
        ip = tokenizer(input_s, return_tensors="pt").input_ids
        lb = tokenizer(label_s, return_tensors="pt").input_ids
        ip=ip.to(device)
        lb=lb.to(device)

        loss = model(input_ids = ip, labels=lb, return_dict=True).loss
        if index%1000 == 0:
            print(f'Epoch: {_}:{index}, Loss:  {loss.item()}')
        
#         if index% 24 ==0:
#         optimizer.zero_grad()
        loss.backward()
        optimizer.step()
#         
        optimizer.zero_grad()


# In[ ]:


model.to(device)
model.eval()


# ## CAsT 2020 data

# In[ ]:


cast20=pd.read_pickle('cast20-qr.pkl')


# In[ ]:


input_seq = []
labels_seq = []
qids = []
for index, data in cast20.iterrows():
    
    query = ''
    if len(data['history']) > 0:
#         history = [q['raw_utterance'] for q in data['history']]
        query = data['raw_utterance']+ ' <s> '+ ' <s> '.join(history[::-1])
    
        if len(data['title'].strip()) > 0:
            query += ' <s> ' + data['title'].strip()
            
        if len(data['description'].strip()) > 0:
            query += ' <s> ' + data['description'].strip()
        
    else:
        query = data['raw_utterance']
    qids.append(data['qid'])
    input_seq.append(query)
    labels_seq.append(data['manual_rewritten_utterance'])


# In[ ]:


len(qids), len(input_seq), len(labels_seq)


# In[ ]:


input_seq[20], labels_seq[20]


# In[ ]:


qrs = []
for i, (qid, t) in enumerate(tqdm(zip(qids, input_seq), total=len(input_seq))):
    tmp={}
    tmp['raw_utterance']=t
    tmp['qid'] = qid
    tmp['manual_rewritten_utterance'] = labels_seq[i]
    
    if qid.endswith('_1'):
        tmp['t5-rewrite']=t 
    else:
        tmp['t5-rewrite'] = qr(t)
    qrs.append(tmp)


# In[ ]:


qrs[10]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




