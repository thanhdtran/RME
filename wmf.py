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




def decompose(train_data, vad_data, weight = 20, num_factors = 100, num_iters = 50, lam = 1e-1, batch_size = 1000):
    #model parameters
    num_factors = num_factors
    num_iters = num_iters
    batch_size = batch_size

    alpha = weight

    n_jobs = 1
    lam_theta = lam_beta = lam
    print '********************** Factorizing using Matrix factorization **********************************'

    S = content_wmf.linear_surplus_confidence_matrix(train_data, alpha=alpha)

    U, V, vad_ndcg = content_wmf.factorize(S, num_factors, vad_data=vad_data, num_iters=num_iters,
                                           init_std=0.01, lambda_U_reg=lam_theta, lambda_V_reg=lam_beta,
                                           dtype='float32', random_state=98765, verbose=True,
                                           recompute_factors=batched_inv_joblib.recompute_factors_batched,
                                           batch_size=batch_size, n_jobs=n_jobs)
    return U, V
