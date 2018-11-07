import itertools
import glob
import os
import sys
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import time
import text_utils
import pandas as pd
from scipy import sparse
from joblib import Parallel, delayed
np.random.seed(98765) #set random seed


import argparse


parser = argparse.ArgumentParser("Description: Running multi-embedding recommendation - RME model")
parser.add_argument('--data_path', default='data', type=str, help='path to the data')
parser.add_argument('--batch_size', default=5000, type=float, help='batch processing')
parser.add_argument('--dataset', default="ml10m", type=str, help='dataset')
args = parser.parse_args()


DATA_DIR =  os.path.join(args.data_path, args.dataset)
unique_uid = list()
with open(os.path.join(DATA_DIR, 'unique_uid.txt'), 'r') as f:
    for line in f:
        unique_uid.append(line.strip())

unique_movieId = list()
with open(os.path.join(DATA_DIR, 'unique_sid.txt'), 'r') as f:
    for line in f:
        unique_movieId.append(line.strip())
n_items = len(unique_movieId)
n_users = len(unique_uid)

print 'number of users:%d ,  number of items: %d'%(n_users, n_items)

def load_data(csv_file, shape=(n_users, n_items)):
    tp = pd.read_csv(csv_file)
    rows, cols = np.array(tp['userId']), np.array(tp['movieId']) #rows will be user ids, cols will be item-ids.
    seq = np.concatenate((  rows[:, None], cols[:, None], np.ones((rows.size, 1), dtype='int')
                          ), axis=1)
    data = sparse.csr_matrix((np.ones_like(rows), (rows, cols)), dtype=np.int16, shape=shape)
    return data, seq, tp





####################Generate item-item co-occurrence matrix based on the user's consumed items history ############
# ##################       This will build a item-item co-occurrence matrix           ############################
#user 1: item 1, item 2, ... item k --> item 1, 2, ..., k will be seen as a sentence ==> do co-occurrence.

def _coord_batch(lo, hi, train_data, prefix = 'item', max_neighbor_words = 100000, choose='macro'):
    rows = []
    cols = []

    for u in xrange(lo, hi):
        #print train_data[u].nonzero()[1] #names all the item ids that the user at index u watched nonzero return a
        # 2D array, index 0 will be the row index and index 1 will be columns whose values are not equal to 0
        lst_words = train_data[u].nonzero()[1]
        if len(lst_words) > max_neighbor_words:
            if choose == 'micro':
                #approach 1: randomly select max_neighbor_words for each word.
                for w in lst_words:
                    tmp = lst_words.remove(w)
                    #random choose max_neigbor words in the list:
                    neighbors = np.random.choice(tmp, max_neighbor_words, replace=False)
                    for c in neighbors:
                        rows.append(w)
                        cols.append(c)
            if choose == 'macro':
                #approach 2: randomly select the sentence with length of max_neigbor_words + 1, then do permutation.
                lst_words = np.random.choice(lst_words, max_neighbor_words + 1, replace=False)
                for w, c in itertools.permutations(lst_words, 2):
                    rows.append(w)
                    cols.append(c)
        else:
            for w, c in itertools.permutations(lst_words, 2):
                rows.append(w)
                cols.append(c)
    if not os.path.exists(os.path.join(DATA_DIR, 'co-temp')): os.mkdir(os.path.join(DATA_DIR, 'co-temp'))
    np.save(os.path.join(DATA_DIR, 'co-temp' ,'%s_coo_%d_%d.npy' % (prefix, lo, hi)),
            np.concatenate([np.array(rows)[:, None], np.array(cols)[:, None]], axis=1)) #append column wise.
    pass


batch_size = args.batch_size

train_data, train_raw, train_df = load_data(os.path.join(DATA_DIR, 'train.csv'))
#clear the co-temp folder:
if os.path.exists(os.path.join(DATA_DIR, 'co-temp')):
    for f in glob.glob(os.path.join(DATA_DIR, 'co-temp', '*.npy')):
        os.remove(f)

GENERATE_ITEM_ITEM_COOCCURENCE_FILE = True
if GENERATE_ITEM_ITEM_COOCCURENCE_FILE:
    t1 = time.time()
    print 'Generating item item co-occurrence matrix'
    start_idx = range(0, n_users, batch_size)
    end_idx = start_idx[1:] + [n_users]
    Parallel(n_jobs=1)(delayed(_coord_batch)(lo, hi, train_data, prefix = 'item', max_neighbor_words = 200) for lo, hi in zip(start_idx, end_idx))
    t2 = time.time()
    print 'Time : %d seconds'%(t2-t1)
    pass
########################################################################################################################
####################Generate user-user co-occurrence matrix based on the same items they backed######################
#####################        This will build a user-user co-occurrence matrix ##########################################

GENERATE_USER_USER_COOCCURENCE_FILE = True
if GENERATE_USER_USER_COOCCURENCE_FILE:
    t1 = time.time()
    print 'Generating user user co-occurrence matrix'
    start_idx = range(0, n_items, batch_size)
    end_idx = start_idx[1:] + [n_items]
    Parallel(n_jobs=8)(delayed(_coord_batch)(lo, hi, train_data.T, prefix = 'user', max_neighbor_words=100) for lo, hi in zip(start_idx, end_idx))
    t2 = time.time()
    print 'Time : %d seconds'%(t2 - t1)
    pass
########################################################################################################################

def _load_coord_matrix(start_idx, end_idx, nrow, ncol, prefix = 'item'):
    X = sparse.csr_matrix((nrow, ncol), dtype='float32')

    for lo, hi in zip(start_idx, end_idx):
        coords = np.load(os.path.join(DATA_DIR, 'co-temp', '%s_coo_%d_%d.npy' % (prefix, lo, hi)))

        rows = coords[:, 0]
        cols = coords[:, 1]

        tmp = sparse.coo_matrix((np.ones_like(rows), (rows, cols)), shape=(nrow, ncol), dtype='float32').tocsr()
        X = X + tmp

        print("%s %d to %d finished" % (prefix, lo, hi))
        sys.stdout.flush()
    return X

BOOLEAN_LOAD_PP_COOCC_FROM_FILE = True
X, Y = None, None
if BOOLEAN_LOAD_PP_COOCC_FROM_FILE:
    print 'Loading item item co-occurrence matrix'
    t1 = time.time()
    start_idx = range(0, n_users, batch_size)
    end_idx = start_idx[1:] + [n_users]
    X = _load_coord_matrix(start_idx, end_idx, n_items, n_items, prefix = 'item') #item item co-occurrence matrix
    print 'dumping matrix ...'
    text_utils.save_pickle(X, os.path.join(DATA_DIR,'item_item_cooc.dat'))
    t2 = time.time()
    print 'Time : %d seconds'%(t2-t1)
else:
    print 'test loading model from pickle file'
    t1 = time.time()
    X = text_utils.load_pickle(os.path.join(DATA_DIR,'item_item_cooc.dat'))
    t2 = time.time()
    print '[INFO]: sparse matrix size of item-item co-occurrence matrix: %d mb\n' % (
                                                    (X.data.nbytes + X.indices.nbytes + X.indptr.nbytes) / (1024 * 1024))
    print 'Time : %d seconds'%(t2-t1)

#X = None
BOOLEAN_LOAD_UU_COOCC_FROM_FILE = True
if BOOLEAN_LOAD_UU_COOCC_FROM_FILE:
    print 'Loading user user co-occurrence matrix'
    t1 = time.time()
    start_idx = range(0, n_items, batch_size)
    end_idx = start_idx[1:] + [n_items]
    Y = _load_coord_matrix(start_idx, end_idx, n_users, n_users, prefix = 'user') #user user co-occurrence matrix

    t2 = time.time()
    print 'Time : %d seconds' % (t2 - t1)

    print 'dumping matrix ...'
    t1 = time.time()
    text_utils.save_pickle(Y, os.path.join(DATA_DIR, 'user_user_cooc.dat'))
    t2 = time.time()
    print 'Time : %d seconds'%(t2-t1)
else:
    print 'test loading model from pickle file'
    t1 = time.time()
    Y = text_utils.load_pickle(os.path.join(DATA_DIR, 'user_user_cooc.dat'))
    t2 = time.time()
    print '[INFO]: sparse matrix size of user user co-occurrence matrix: %d mb\n' % (
                                                    (Y.data.nbytes + Y.indices.nbytes + Y.indptr.nbytes) / (1024 * 1024))
    print 'Time : %d seconds'%(t2-t1)
