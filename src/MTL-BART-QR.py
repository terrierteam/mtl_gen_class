#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from transformers import  BartTokenizer


# In[ ]:


from bart.modeling_bart import MTLBart#BartForSequenceClassification#MTLBart
from bart.configuration_bart import BartConfig


# In[ ]:


BartConfig.num_labels = 2


# In[ ]:


def set_seed(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

set_seed(42)

model = MTLBart.from_pretrained('facebook/bart-base')


# In[ ]:


tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')


# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print ("device ",device)
model = model.to(device)


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


castur_cast19.head(1)


# In[ ]:


input_seq = []
labels_seq = []

for index, row in castur_cast19.iterrows():
    history = row['prev_qs'][::-1]
    input_seq.append(row['candidate']+ ' </s> '+ ' </s> '.join(history))
    labels_seq.append([row['qr'], row['label']])


# In[ ]:


len(input_seq), len(labels_seq)


# In[ ]:


lbs = [l[0] for l in labels_seq]
lbs_tps = [l[1] for l in labels_seq]
len(lbs_tps), len(lbs)


# In[ ]:


input_seq[0]


# In[ ]:


tokenizer(input_seq[0], return_tensors="pt")


# In[ ]:


tokenizer.eos_token


# In[ ]:


for _ in range(10):
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


# In[ ]:


def qr(sentence):
    model.eval()
    max_len = 256
    
    text =  sentence + " </s>"
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


castur_cast19_eval = pd.read_pickle('cast19/castur_cast_eval.pkl')
castur_cast19_eval.head(2)


# In[ ]:


input_seq_eval = []
turn_number = []
# labels_seq_eval = []
for index, row in castur_cast19_eval.iterrows():
#     if row['turn_number'].endswith('_1'):
#         continue
    turn_number.append(row['turn_number'])
    history = row['prev_qs'][::-1]
    input_seq_eval.append(row['candidate']+ ' </s> '+ ' </s> '.join(history))
   


# In[ ]:


res_eval = {}
for tnum, sent in zip(turn_number,input_seq_eval):
    if tnum.endswith('_1'):
        res_eval[tnum] = sent.replace('</s>','').strip()
    else:
        res_eval[tnum] = qr(sent)


# In[ ]:


res_eval


# In[ ]:


cast19_eval = pd.read_pickle('cast19/castur_cast_eval_with_bart_qr.pkl')


# In[ ]:


cast19_eval.head()


# In[ ]:


cast19_eval['bart_mtl_qr'] = castur_cast19_eval.apply(lambda row: res_eval[row['turn_number']] ,axis=1)


# In[ ]:


cast19_eval.head()


# In[ ]:


cast19_eval.to_pickle('cast19/castur_cast_eval_with_bart_qr_with_bart_mtl.pkl')


# In[ ]:




