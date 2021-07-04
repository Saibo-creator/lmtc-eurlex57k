## Large-Scale Multi-Label Text Classification on EU Legislation

This is the code used for the first downstream task, it's derived from the original code for the following paper:

> I. Chalkidis, M. Fergadiotis, P. Malakasiotis and I. Androutsopoulos, "Large-Scale Multi-Label Text Classification on EU Legislation". Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL 2019), Florence, Italy, (short papers), 2019 (https://www.aclweb.org/anthology/P19-1636/)

The original code is: https://github.com/iliaschalkidis/lmtc-eurlex57k

Major modifications on the original code:
1. add support for tensorflow 2(integrate keras api)
2. integrate huggingface API 
3. remove models not based on transformers

## Conda Environment:

```shell
conda env create -f huggingface_lmtc.yml
```
## Quick start:

### Install python requirements:

```
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python
>>> import nltk
>>> nltk.download('punkt')
```

### Download dataset (EURLEX57K):

```
wget -O data/datasets/datasets.zip http://nlp.cs.aueb.gr/software_and_datasets/EURLEX57K/datasets.zip
unzip data/datasets/datasets.zip -d data/datasets/EURLEX57K
rm data/datasets/datasets.zip
rm -rf data/datasets/EURLEX57K/__MACOSX
mv data/datasets/EURLEX57K/dataset/* data/datasets/EURLEX57K/
rm -rf data/datasets/EURLEX57K/dataset
wget -O data/datasets/EURLEX57K/EURLEX57K.json http://nlp.cs.aueb.gr/software_and_datasets/EURLEX57K/eurovoc_en.json
```

### Select training options from the configuration JSON file:

E.g., fine-tune a legal-roberta-base:

```
{
  "task": {
    "dataset": "EURLEX57K",
    "decision_type": "multi_label"
  },
  "model": {
    "architecture": "transformer",
    "dropout_rate": 0.2,
    "lr": 1e-5,
    "batch_size": 4,
    "epochs": 40,
    "freeze_pretrained": false,
    "uri": "saibo/legal-roberta-base"
  },
  "sampling": {
    "max_sequences_size": null,
    "max_sequence_size": 128,
    "max_label_size": 15,
    "few_threshold": 50,
    "hierarchical": false,
    "evaluation@k": 10
  }
}
```

**Supported models:** Most models available on huggingface
**Important models:** 'saibo/legal-roberta-base', 'roberta-base', 'bert-base-uncased', 'nlpaueb/legal-bert-base-uncased'


### Train a model:

```
python lmtc.py
```
