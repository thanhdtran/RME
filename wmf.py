import itertools
import os
import sys
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import pandas as pd
from scipy import sparse

import content_wmf
import batched_inv_joblib
import rec_eval
import glob



# def load_data(csv_file, shape=(0, 0)):
#     tp = pd.read_csv(csv_file)
#     count, rows, cols = np.array(tp['count']), np.array(tp['uid']), np.array(tp['sid']) #rows will be user ids, cols will be projects-ids.
#     seq = np.concatenate((  rows[:, None], cols[:, None], np.ones((rows.size, 1), dtype='int'),
#                             count[:, None]
#                           ), axis=1)
#     data = sparse.csr_matrix((np.ones_like(rows), (rows, cols)), dtype=np.int16, shape=shape)
#     return data, seq, tp

def decompose(train_data, vad_data, weight = 20, num_factors = 100, num_iters = 50, lam = 1e-1, batch_size = 1000):
    # DEBUG_MODE = False
    # DATA_DIR = 'data/rec_data/all'
    # if DEBUG_MODE:
    #     DATA_DIR = 'data/rec_data/debug'
    # unique_uid = list()
    # with open(os.path.join(DATA_DIR, 'unique_uid_sub.txt'), 'r') as f:
    #     for line in f:
    #         unique_uid.append(line.strip())
    #
    # unique_sid = list()
    # with open(os.path.join(DATA_DIR, 'unique_sid_sub.txt'), 'r') as f:
    #     for line in f:
    #         unique_sid.append(line.strip())
    # n_projects = len(unique_sid)
    # n_users = len(unique_uid)

    #model parameters
    num_factors = num_factors
    num_iters = num_iters
    batch_size = batch_size

    alpha = weight

    n_jobs = 1
    lam_theta = lam_beta = lam
    print '********************** Factorizing using Matrix factorization **********************************'

    # print '********************** FOLD %d **********************************'%FOLD
    # vad_data, vad_raw, vad_df = load_data(os.path.join(DATA_DIR, 'vad.num.sub.fold%d.csv'%FOLD), shape=(n_users, n_projects))
    # test_data, test_raw, test_df = load_data(os.path.join(DATA_DIR, 'test.num.sub.fold%d.csv'%FOLD), shape=(n_users, n_projects))
    # train_data, train_raw, train_df =  load_data(os.path.join(DATA_DIR, 'train.num.sub.fold%d.csv'%FOLD), shape=(n_users, n_projects))


    S = content_wmf.linear_surplus_confidence_matrix(train_data, alpha=alpha)

    U, V, vad_ndcg = content_wmf.factorize(S, num_factors, vad_data=vad_data, num_iters=num_iters,
                                           init_std=0.01, lambda_U_reg=lam_theta, lambda_V_reg=lam_beta,
                                           dtype='float32', random_state=98765, verbose=True,
                                           recompute_factors=batched_inv_joblib.recompute_factors_batched,
                                           batch_size=batch_size, n_jobs=n_jobs)
    return U, V
