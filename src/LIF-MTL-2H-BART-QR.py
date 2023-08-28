#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from transformers import  BartTokenizer


# In[2]:


from bart.modeling_bart import MTLBart#BartForSequenceClassification#MTLBart
from bart.configuration_bart import BartConfig


# In[3]:


BartConfig.num_labels = 2


# In[4]:


def set_seed(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

set_seed(42)

model = MTLBart.from_pretrained('facebook/bart-base')


# In[5]:


tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')


# In[6]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print ("device ",device)
model = model.to(device)


# In[7]:


LEARNING_RATE = 1e-4
optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)


# In[8]:


from tqdm.notebook import tqdm


# In[9]:


import pandas as pd


# In[10]:


tokenizer.add_tokens('<sep>')
model.resize_token_embeddings(len(tokenizer))
model.to(device)


# In[11]:


train = pd.read_pickle('data-lif-canard/train_canard_lif.pkl')


# In[12]:


train.head(1)


# In[13]:


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
    
    

    input_seq.append(input_s)
#     labels_seq.append(row['qr'])
#     labels_seq.append([row['qr'], row['label']])
#     if row['label']==0:
    labels_seq.append([row['qr'], row['label']])
#     else:
#         labels_seq.append('shift '+ row['qr'])


# In[ ]:


# input_seq = []
# labels_seq = []

# for index, row in castur_cast19.iterrows():
#     history = row['prev_qs'][::-1]
#     input_seq.append(row['candidate']+ ' </s> '+ ' </s> '.join(history))
#     labels_seq.append([row['qr'], row['label']])


# In[14]:


len(input_seq), len(labels_seq)


# In[15]:


lbs = [l[0] for l in labels_seq]
lbs_tps = [l[1] for l in labels_seq]
len(lbs_tps), len(lbs)


# In[16]:


input_seq[0]


# In[17]:


tokenizer(input_seq[0], return_tensors="pt")


# In[18]:


input_seq[0:3], labels_seq[0:3]


# In[ ]:


for _ in range(5):
    for index, (input_s, label_s) in enumerate(tqdm(zip(input_seq, lbs), total=len(input_seq))):
        model.train()
        ip = tokenizer(input_s, return_tensors="pt").input_ids
         
        ip_att = tokenizer(input_s, return_tensors="pt").attention_mask
        lb = tokenizer(label_s, return_tensors="pt").input_ids
        ip=ip.to(device)
        lb=lb.to(device)
        ip_att=ip_att.to(device)
        lb_tps = lbs_tps[index]
        lb_tps=torch.tensor(lb_tps).to(device)

        #outputs = model(input_ids = ip, attention_mask=ip_att, labels=lb_tps, return_dict=False) # labels_topic_shift=lb_tps,
        outputs = model(input_ids = ip, attention_mask=ip_att, labels=lb, labels_topic_shift=lb_tps,return_dict=False) # labels_topic_shift=lb_tps,
        
        if len(outputs)==5:
            lm_loss,loss, logits, lm_logits, tmp_ = outputs
        else:
            raise ValueError("cannot learn as MTL")
            
        if index%1000 == 0:
            print(f'Epoch: {_}:{index}, Loss:  {loss.item()}, Loss LM:  {lm_loss.item()}')
        
        total_loss = loss + lm_loss

#         loss.backward()
#         lm_loss.backward()
        total_loss.backward()
    
        optimizer.step()
#         
        optimizer.zero_grad()


# In[ ]:


# for _ in range(5):
#     for index, (input_s, label_s) in enumerate(tqdm(zip(input_seq, lbs))):
#         model.train()
#         ip = tokenizer(input_s, return_tensors="pt").input_ids
         
#         ip_att = tokenizer(input_s, return_tensors="pt").attention_mask
#         lb = tokenizer(label_s, return_tensors="pt").input_ids
#         ip=ip.to(device)
#         lb=lb.to(device)
#         ip_att=ip_att.to(device)
#         lb_tps = lbs_tps[index]
#         lb_tps=torch.tensor(lb_tps).to(device)

#         outputs = model(input_ids = ip, attention_mask=ip_att, labels=lb_tps, return_dict=False) # labels_topic_shift=lb_tps,
# #         outputs = model(input_ids = ip, attention_mask=ip_att, labels=lb, labels_topic_shift=lb_tps,return_dict=False) # labels_topic_shift=lb_tps,
#         if len(outputs)==4:
#             lm_loss,loss, logits, lm_logits = outputs
#         else:
#             raise ValueError("cannot learn as MTL")
            
#         if index%1000 == 0:
#             print(f'Epoch: {_}:{index}, Loss:  {loss.item()}')
        
# #         if index% 24 ==0:
# #         optimizer.zero_grad()
#         loss.backward()
#         lm_loss.backward()
        
#         optimizer.step()
# #         
#         optimizer.zero_grad()


# In[41]:


def qr(sentence):
    model.eval()
    max_len = 256
    
    text =  "paraphrase: " + sentence + " </s>"
#     text =  sentence #+ " </s>"
    encoding = tokenizer.encode_plus(text,pad_to_max_length=False, return_tensors="pt")
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


# In[42]:


df_test= pd.read_pickle('data-lif-canard/topics_lif_with_t5_mtl.pkl')


# In[43]:


df_test.head(2)


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
    for ht in row['history']:
        history.append(ht)
    
    input_s += ' <sep> '+ ' <sep> '.join(history[::-1])
        
    input_seq_eval.append(input_s)


# In[45]:


len(input_seq_eval), len(qids)


# In[46]:


input_seq_eval[0:2]


# In[47]:


res_eval = {}
for tnum, sent in tqdm(zip(qids,input_seq_eval), total=len(qids)):
#     if tnum.endswith('_1'):
#         res_eval[tnum] = sent.replace('<extra_id_0>','').strip()
#     else:
    res_eval[tnum] = qr(sent)


# In[39]:


res_eval_tmp = res_eval.copy()


# In[40]:


res_eval


# In[ ]:





# In[ ]:




