# Multi-Task Learning of Query Generation and Classification for Generative Conversational Question Rewriting

This repository contains the code and datasets for the EMNLP 2023 paper "Multi-Task Learning of Query Generation and Classification for Generative Conversational Question Rewriting". The paper proposes a novel multi-task learning approach to rewrite ambiguous conversational questions into well-defined queries while also identifying topic continuity. The models, based on BART and T5 architectures, demonstrate significant performance improvements over single-task baselines.

Install dependencies:

```bash
git clone
cd mtl_gen_class
pip install -r requirements.txt
```

## Data Preparation

By default, we expect raw and processed data to be stored in `./data/` :

### OR-QuAC

#### OR-QuAC files download

Download necessary OR-QuAC files and store them into `./data/or-quac`:

```bash
mkdir data/or-quac
cd data/or-quac
wget https://ciir.cs.umass.edu/downloads/ORConvQA/all_blocks.txt.gz
wget https://ciir.cs.umass.edu/downloads/ORConvQA/qrels.txt.gz
gzip -d *.txt.gz
mkdir preprocessed
cd preprocessed
wget https://ciir.cs.umass.edu/downloads/ORConvQA/preprocessed/train.txt
wget https://ciir.cs.umass.edu/downloads/ORConvQA/preprocessed/test.txt
wget https://ciir.cs.umass.edu/downloads/ORConvQA/preprocessed/dev.txt


