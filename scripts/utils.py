import numpy as np
import random
from collections import defaultdict
from keras.preprocessing.sequence import pad_sequences
import os
os.environ['KMP_WARNINGS'] = '0'
def sample(from_array, n_sample, replace=True, p=None):
    x = np.asarray(from_array)
    lenx = len(x)
    ind_range = np.arange(lenx)
    if lenx < n_sample:
        if replace:
            sampled_inds = np.random.choice(ind_range, n_sample, replace=True, p=p)
        else:
            sampled_inds = np.random.choice(ind_range, lenx, replace=True, p=p)
    else:
        sampled_inds = np.random.choice(ind_range, n_sample, replace=False, p=p)

    samples = x[sampled_inds]
    return samples.tolist()


def expand_items(dict):
    expanded = []
    items = dict.items()
    for key, vlist in items:
        expanded += [(key, x) for x in vlist]
    return expanded

def texts_to_indices(X, max_len, tokenizer):
    X = [tokenizer.encode(sent, add_special_tokens=True) for sent in X]
    X = pad_sequences(X, maxlen=max_len, padding='post', truncating='post').tolist()
    return X

def calc_acc(preds, targets):
    return np.mean(np.asarray(preds) == np.asarray(targets))






