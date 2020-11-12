import json
import re
import time
import tempfile
import glob
import tqdm

import pickle
import lmtc
import numpy as np
from copy import deepcopy
from collections import Counter
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import Sequence
from nltk.tokenize import word_tokenize
from sklearn.metrics import f1_score, precision_score, recall_score


from json_loader import JSONLoader
from vectorizer import W2VVectorizer, ELMoVectorizer, BERTVectorizer
from data import DATA_SET_DIR, MODELS_DIR
from configuration import Configuration
from metrics import mean_recall_k, mean_precision_k, mean_ndcg_score, mean_rprecision_k
from neural_networks.lmtc_networks.document_classification import DocumentClassification
from neural_networks.lmtc_networks.label_driven_classification import LabelDrivenClassification






if __name__ == '__main__':
    import argparse, sys

    parser = argparse.ArgumentParser()
    #首先是mandatory parameters
    parser.add_argument("model_name", help="legalBert vs roberta",choices=["legalBert","roberta"])# 必须以 python script.py mandatory_para_value....(如果没有这个参数会报错)

    # #然后是optional parameters，以--或者-开头
    # parser.add_argument('-o', '--output', type=argparse.FileType('w'), default=sys.stdout,
    #                     help='Output file (default stdout).')
    # parser.add_argument('-n', '--num-seeds', default=10, type=int, help='Number of seeds to generate.')
    # parser.add_argument('-w', '--ngram-range', nargs=2, type=int, default=(3, 3), help='Number of words per seed.')
    # parser.add_argument('-p', '--min-proba', default=0.95, help='Min. GSW probability for a seed to be accepted.')
    # parser.add_argument('--quiet', action='store_true', help='Be quiet')


    #解析参数
    args = parser.parse_args()

    model_name=args.model_name

    Configuration.configure()

    lmtc=lmtc.LMTC()


    network = LabelDrivenClassification(lmtc.label_terms_ids)



    with open("/mnt/localdata/geng/model/lmtc_models/downstream/multiLabelClassification/{0}/prediction.pickle".format(model_name), "rb") as f:
        pred_labels,true_labels=pickle.load(f)

    pred_labels=np.array(pred_labels)
    true_labels=np.array(true_labels)
    lmtc.calculate_performance(true_labels,pred_labels)





