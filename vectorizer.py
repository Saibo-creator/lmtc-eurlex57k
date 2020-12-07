from typing import List
import numpy as np
from configuration import Configuration
from transformers import AutoTokenizer
import pdb





class HgBERTVectorizer():

    def __init__(self):
        super().__init__()
        self.uri=Configuration["model"]["uri"]

    def load_tokenizer(self,max_sequence_size):
        hg_tokenizer = AutoTokenizer.from_pretrained(self.uri,max_len=max_sequence_size)
        
        return hg_tokenizer

    def produce_label_term_ids(self, sequences: List[List[str]], max_sequence_size=100, **kwargs):

        bert_tokenizer=self.load_tokenizer(max_sequence_size)
        print("Loaded tokenizer from ",self.uri)
        token_indices = np.zeros((len(sequences), max_sequence_size), dtype=np.int32)
        seg_indices = np.zeros((len(sequences), max_sequence_size), dtype=np.int32)
        mask_indices = np.zeros((len(sequences), max_sequence_size), dtype=np.int32)

        for i, tokens in enumerate(sequences):
            bpes = bert_tokenizer.encode(' '.join([token for token in tokens]),padding='max_length', truncation=True, max_length=max_sequence_size)
            limit = min(max_sequence_size, len(bpes))
            token_indices[i][:limit] = bpes[:limit]
            mask_indices[i][:limit] = np.ones((limit,), dtype=np.int32)

        result = np.concatenate((np.reshape(token_indices, [len(sequences), max_sequence_size, 1]),
                               np.reshape(mask_indices, [len(sequences), max_sequence_size, 1]),
                               np.reshape(seg_indices, [len(sequences), max_sequence_size, 1])), axis=-1)

        return result

    def vectorize_inputs(self, sequences: List[List[str]], max_sequence_size=100, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(self.uri, max_len=max_sequence_size)

        sequences=[' '.join([token for token in tokens]) for tokens in sequences]
        
        index_plus=tokenizer(sequences,return_tensors="tf", padding='max_length', truncation=True, max_length=max_sequence_size)

        return index_plus
