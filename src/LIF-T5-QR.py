#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import transformers
transformers.__version__


# In[ ]:


# !pip install -U transformers==3.1.0


# In[ ]:


get_ipython().system('nvidia-smi')


# In[1]:


import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer
# from transformers import BartForConditionalGeneration, BartTokenizer


def set_seed(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

set_seed(42)

model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_paraphraser')


# In[2]:


tokenizer = T5Tokenizer.from_pretrained('t5-base')


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


train = pd.read_pickle('data-lif-canard/train_canard_lif.pkl')


# In[ ]:


# special_tokens_dict = {'additional_special_tokens': ['<extra_id_0>','<extra_id_1>']}

# tokenizer.add_special_tokens(special_tokens_dict)


# In[8]:


train.head(1)


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
    
    input_s += ' <extra_id_0> '+ ' <extra_id_1> '.join(history)
    
    
    
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


# In[10]:


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


# In[11]:


input_seq[0:3], labels_seq[0:3]


# In[ ]:


tokenizer(input_seq[0], return_tensors="pt")


# In[ ]:


# lbs = [l[0] for l in labels_seq]
# lbs_tps = [l[1] for l in labels_seq]
# len(lbs_tps), len(lbs)


# In[ ]:


class LCDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, seqs, labels):
        self.labels = labels
        self.seqs = seqs
#         self.inputs =[]
#         self.atts = []
#         self.targets = []
#         self.build()
        
#     def build(self):
#         for seq in tqdm(self.seqs):
#             tk = tokenizer(seq, return_tensors="pt")
#             self.inputs.append(tk.input_ids)
#             self.atts.append(tk.attention_mask)
        
#         self.targets = [tokenizer(lb, return_tensors="pt").input_ids for lb in self.labels]
        
    def __len__(self):
        return len(self.labels)
    def __getitem__(self,index):
        return {'seq':self.seqs[index], 'label':self.labels[index]}


# In[ ]:


params = {'batch_size': 4,
          'shuffle': True,
          'num_workers': 6}


# In[ ]:


training_set = LCDataset(tokenizer, input_seq, labels_seq)
training_generator = torch.utils.data.DataLoader(training_set, **params)


# In[12]:


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


# In[13]:


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


# In[14]:


df_test= pd.read_pickle('data-lif-canard/test-retrieval.pkl')


# In[15]:


df_test.head(2)


# In[ ]:





# In[16]:


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
    
    input_s += ' <extra_id_0> '+ ' <extra_id_1> '.join(history[::-1])
        
    input_seq_eval.append(input_s)


# In[17]:


len(input_seq_eval), len(qids)


# In[18]:


input_seq_eval[0:2]


# In[19]:


res_eval = {}
for tnum, sent in tqdm(zip(qids,input_seq_eval), total=len(qids)):
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


# cast19_eval = pd.read_pickle('cast19/castur_cast_eval_with_bart_qr_with_bart_mtl.pkl')


# In[ ]:


# cast19_eval.head(2)


# In[20]:


df_test_t5_mtl= pd.read_pickle('data-lif-canard/topics_lif_with_t5_mtl.pkl')


# In[21]:


df_test_t5_mtl['t5_qr'] = df_test_t5_mtl.apply(lambda row: res_eval[row['qid']] ,axis=1)


# In[22]:


# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 100)
pd.set_option("display.max_colwidth", 200)


# In[24]:


df_test_t5_mtl.sample(5)


# In[25]:


df_test_t5_mtl.to_pickle('data-lif-canard/topics_lif_with_t5_mtl.pkl')


# In[26]:


get_ipython().system('mkdir t5-model')


# In[28]:


torch.save(model,'t5-model/STL-QR')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




