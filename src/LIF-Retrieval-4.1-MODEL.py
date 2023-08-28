#!/usr/bin/env python
# coding: utf-8

# In[26]:


import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64/"
# import pyterrier as pt
# if not pt.started():
#     pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"], mem=8000)
import pyterrier as pt
if not pt.started():
    pt.init(mem=32000)
#set pyterrier's run cache directory to be in your /users/tr.$USER space. Adjust if necessary
pt.cache.CACHE_DIR = "/nfs/pyterrier_transformer_cache" 


# In[2]:


import pickle,json
import pandas as pd


# In[3]:


pd.set_option('display.max_column',None)
pd.set_option('display.max_rows',None)
pd.set_option('display.max_seq_items',None)
pd.set_option('display.max_colwidth', 500)
pd.set_option('expand_frame_repr', True)


# In[4]:


import re
def cleanStr(text):
#     text = q["query"]
    text = text.replace('\W', ' ')
    text = text.replace('?', '')
    text = text.replace("á", 'a')
    text = text.replace("é", 'e')
    text = text.replace("ö", 'o')
    text = text.replace("Č", 'C')
    text = text.replace("ć", 'c')
    text = text.replace("ó", 'o')
    text = text.replace("ă", 'a')
    text = text.replace("ä", 'a')
    text = text.replace("ü", 'u')
    text = text.replace("ā", 'a')
    text = text.replace("í", 'i')
    text = text.replace("ÿ", 'y')
    text = re.sub('[^0-9a-zA-Z]+', ' ', text)
    return text
def cleanStrDF(q):
    text = q["query"]
    text = text.replace('\W', ' ')
    text = text.replace('?', '')
    text = text.replace("á", 'a')
    text = text.replace("é", 'e')
    text = text.replace("ö", 'o')
    text = text.replace("Č", 'C')
    text = text.replace("ć", 'c')
    text = text.replace("ó", 'o')
    text = text.replace("ă", 'a')
    text = text.replace("ä", 'a')
    text = text.replace("ü", 'u')
    text = text.replace("ā", 'a')
    text = text.replace("í", 'i')
    text = text.replace("ÿ", 'y')
    text = re.sub('[^0-9a-zA-Z]+', ' ', text)
    return text


# In[44]:


def appendall(query, history, turn=-1):
    return query + ' ' + ' '.join(history)
class MixHistory(TransformerBase):
    def __init__(self, inner_pipe, func, history=-1):
        self.inner_pipe = inner_pipe
        self.history = history
        self.func = func
    def transform(self, topics):
        topicsNew = topics.copy()
        topicsNew['query'] = topicsNew.apply(lambda row: self.func(row['query'], row['history'], self.history), axis=1)
        topicsNew["query"] = topicsNew["query"].apply(cleanStr)
        res = self.inner_pipe.transform(topicsNew)
        return res


# In[5]:


index_path = "./index_orquac/data.properties"
index = pt.IndexFactory.of(index_path)


# In[6]:


eval_metrics = ["map", "num_rel_ret", "ndcg", "ndcg_cut_10", "recip_rank", "recall_5", "recall_10", "recall_1000"]


# In[30]:


topics = pd.read_pickle('data-lif-canard/topics_lif_with_t5mtl-bartmtl.pkl')


# In[31]:


topics.sample(5)


# In[32]:


from pyterrier.transformer import TransformerBase


# In[33]:


from pyterrier.rewrite import RM3, AxiomaticQE

firstpass_bm25 = pt.BatchRetrieve(index, wmodel="BM25", 
                                 verbose=True, metadata=["docno"])

firstpass_dph = pt.BatchRetrieve(index, wmodel="DPH", 
                                 verbose=True, metadata=["docno"])


# In[34]:


appendall_bm25  = MixHistory(firstpass_bm25, appendall)
appendall_dph  = MixHistory(firstpass_dph, appendall)


# In[35]:


bm25_clean = pt.apply.query(cleanStrDF) >> firstpass_bm25
dph_clean = pt.apply.query(cleanStrDF) >> firstpass_dph


# In[36]:


from pyterrier.transformer import TransformerBase
class RewriteQuery(TransformerBase):
    def __init__(self, inner_pipe, rewritename):
        self.inner_pipe = inner_pipe
        self.rewritename = rewritename
    def transform(self, topics):
        topicsNew = topics.copy()
        topicsNew["query"] = topicsNew[self.rewritename]
        res = self.inner_pipe.transform(topicsNew)
#         res['text'] = res.apply(lambda row: docnototext[row['docno']], axis=1)
#         res['docno'] = res['doc_no']
        return res


# In[37]:


raw_bm25 = RewriteQuery(bm25_clean, 'query')
raw_dph = RewriteQuery(dph_clean, 'query')
qr_bm25 = RewriteQuery(bm25_clean, 'qr')
qr_dph = RewriteQuery(dph_clean, 'qr')


# In[38]:


t5_mtl_bm25 = RewriteQuery(bm25_clean, 't5mtl_qr')
t5_mtl_dph = RewriteQuery(dph_clean, 't5mtl_qr')
t5_stl_bm25 = RewriteQuery(bm25_clean, 't5_qr')
t5_stl_dph = RewriteQuery(dph_clean, 't5_qr')


# In[47]:


t5_mtl_tmp_dph = RewriteQuery(dph_clean, 't5mtl-tmp2_qr')
t5_mtl_tmp_bm25 = RewriteQuery(bm25_clean, 't5mtl-tmp2_qr')


# In[40]:


bart_mtl_bm25 = RewriteQuery(bm25_clean, 'bartmtl_qr')
bart_mtl_dph = RewriteQuery(dph_clean, 'bartmtl_qr')
bart_stl_bm25 = RewriteQuery(bm25_clean, 'bartstl_qr')
bart_stl_dph = RewriteQuery(dph_clean, 'bartstl_qr')
bart_mtl2_bm25 = RewriteQuery(bm25_clean, 'bartmtl2_qr')
bart_mtl2_dph = RewriteQuery(dph_clean, 'bartmtl2_qr')


# In[41]:


bart_2h_mtl2_bm25 = RewriteQuery(bm25_clean, 'bart2hmtl_qr')
bart_2h_mtl2_dph = RewriteQuery(dph_clean, 'bart2hmtl_qr')


# In[42]:


json_qrels = json.load(open('data/qrels.txt', 'r'))


# In[43]:


all_qrels = []
for k,v in json_qrels.items():
    for i,j in v.items():
        tmp = {}
        tmp['qid']=k
        tmp['iter']=0
        tmp['docno']=i
        tmp['label']=j
        all_qrels.append(tmp)


# In[44]:


qrels = pd.DataFrame(all_qrels)


# In[48]:


pt.Experiment([  t5_mtl_tmp_bm25], 
              topics, qrels, 
              eval_metrics,
              baseline=0, round=4,
              names=["T5" ])


# In[45]:


pt.Experiment([  t5_mtl_tmp_dph], 
              topics, qrels, 
              eval_metrics,
              baseline=0, round=4,
              names=["T5" ])


# In[46]:


pt.Experiment([  appendall_bm25,  appendall_dph], 
              topics, qrels, 
              eval_metrics,
              baseline=1, round=4,
              names=["Append all BM25" ,"Append all DPH"])


# In[43]:


pt.Experiment([  bart_mtl_bm25,  bart_mtl_dph,bart_2h_mtl2_bm25 , bart_2h_mtl2_dph], 
              topics, qrels, 
              eval_metrics,
              baseline=1, round=4,
              names=["BART MTL BM25" ,"BART MTL DPH", "BART 2H MTL BM25" ,"BART 2H MTL DPH"])


# In[21]:


pt.Experiment([raw_bm25, raw_dph,qr_bm25,qr_dph, t5_stl_bm25, t5_stl_dph, t5_mtl_bm25, t5_mtl_dph], 
              topics, qrels, 
              eval_metrics,
              baseline=0, round=4,
              names=["Raw BM25","Raw DPH","QR BM25","QR DPH", "T5 STL BM25","T5 STL DPH", "T5 MTL BM25","T5 MTL DPH"])


# In[30]:


pt.Experiment([ t5_stl_bm25, t5_stl_dph, t5_mtl_bm25, t5_mtl_dph], 
              topics, qrels, 
              eval_metrics,
              baseline=0, round=4,
              names=["BART MTL BM25","BART MTL DPH","T5 MTL BM25","T5 MTL DPH"])


# In[35]:


pt.Experiment([ bart_stl_bm25, bart_stl_dph, bart_mtl_bm25, bart_mtl_dph, bart_mtl2_bm25, bart_mtl2_dph], 
              topics, qrels, 
              eval_metrics,
              baseline=0, round=4,
              names=["BART MTL BM25","BART MTL DPH", "BART MTL BM25","BART MTL DPH", "BART MTL2 BM25","BART MTL2 DPH"])


# In[24]:


pt.Experiment([  bart_stl_dph,  bart_mtl_dph], 
              topics, qrels, 
              eval_metrics,
              baseline=1, round=4,
              names=["BART STL DPH" ,"BART MTL DPH"])


# In[25]:


pt.Experiment([  bart_stl_dph,  bart_mtl_dph], 
              topics, qrels, 
              eval_metrics,
              baseline=1, round=4,
              names=["BART STL DPH" ,"BART MTL DPH"])


# In[36]:


pt.Experiment([ qr_bm25, qr_dph, t5_stl_bm25, t5_stl_dph, t5_mtl_bm25, t5_mtl_dph], 
              topics, qrels, 
              eval_metrics,
              baseline=0, round=4,
              names=["QR BM25", "QR DPH", "T5 STL BM25","T5 STL DPH","T5 MTL BM25","T5 MTL DPH"])


# In[51]:


res_t5_stl_dph = t5_stl_dph.transform(topics)
res_t5_mtl_dph = t5_mtl_dph.transform(topics)
res_qr_dph = qr_dph.transform(topics)
res_bart_stl_dph = bart_stl_dph.transform(topics)
res_bart_mtl_dph = bart_mtl2_dph.transform(topics)


# In[52]:


eval_t5_stl_dph = pt.Utils.evaluate(res_t5_stl_dph, qrels=qrels, metrics=['map','recip_rank','recall_1000', 'ndcg'], perquery=True)
eval_t5_mtl_dph = pt.Utils.evaluate(res_t5_mtl_dph, qrels=qrels, metrics=['map','recip_rank','recall_1000', 'ndcg'], perquery=True)
eval_qr_dph = pt.Utils.evaluate(res_qr_dph, qrels=qrels, metrics=['map','recip_rank','recall_1000', 'ndcg'], perquery=True)
eval_bart_stl_dph = pt.Utils.evaluate(res_bart_stl_dph, qrels=qrels, metrics=['map','recip_rank','recall_1000', 'ndcg'], perquery=True)
eval_bart_mtl_dph = pt.Utils.evaluate(res_bart_mtl_dph, qrels=qrels, metrics=['map','recip_rank','recall_1000', 'ndcg'], perquery=True)


# In[ ]:


eval_t5_stl_dph


# In[57]:


res_t5_tmp_dph = t5_mtl_tmp_dph.transform(topics.head(5))


# In[68]:


import ir_measures
from ir_measures import *


# In[70]:


from pyterrier.measures import *


# In[83]:


pt.Experiment(
    [t5_mtl_tmp_dph,t5_mtl_dph],
    topics,
    qrels,
    eval_metrics=['ndcg', 'map', "ndcg_cut_10"],
#                 ^ using ir_measures
)


# In[86]:


pt.Experiment(
    [t5_mtl_tmp_dph],
    topics,
    qrels,
    eval_metrics=['ndcg', 'map', "ndcg_cut_10"],
#                 ^ using ir_measures
)


# In[90]:


pt.Experiment(
    [t5_mtl_tmp_dph],
    topics,
    qrels,
    eval_metrics=['ndcg', 'map', "ndcg_cut_10"],
#                 ^ using ir_measures
)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




