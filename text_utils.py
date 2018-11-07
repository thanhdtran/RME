import re
import cPickle as pickle
import numpy as np
from scipy import sparse
#token_pattern = re.compile(r"\b\w\w+\b", re.U)
token_pattern = re.compile(r"\b[a-zA-Z][a-zA-Z]+\b", re.U)

def custom_tokenizer( s, min_term_length = 2, stopwords = None):
    if (stopwords == None):
        return [x.lower() for x in token_pattern.findall(s) if
                (len(x) >= min_term_length and x[0].isalpha())]
    else:
        return [x.lower() for x in token_pattern.findall(s) if (len(x) >= min_term_length and x[0].isalpha() and x not in stopwords)]

def load_stop_words(inpath = "input/text/stopwords.txt"):
    stopwords = set()
    with open(inpath) as f:
        lines = f.readlines()
        for l in lines:
            l = l.strip()
            if len(l) > 0:
                stopwords.add(l)
    return stopwords

def save_pickle(matrix, filename):
    with open(filename, 'wb') as outfile:
        pickle.dump(matrix, outfile, pickle.HIGHEST_PROTOCOL)
def load_pickle(filename):
    with open(filename, 'rb') as infile:
        matrix = pickle.load(infile)
    return matrix