from joblib import Parallel, delayed
import glob
import os
from model_runner import  ModelRunner
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from scipy import sparse
import numpy as np
import global_constants as gc
import time
import text_utils
import pandas as pd
import argparse


parser = argparse.ArgumentParser("Description: Running multi-embedding recommendation - RME model")
parser.add_argument('--data_path', default='data', type=str, help='path to the data')
parser.add_argument('--saved_model_path', default='MODELS', type=str, help='path to save the optimal learned parameters')
parser.add_argument('--s', default=1, type=int, help='a pre-defined shifted value for measuring SPPMI')
parser.add_argument('--model', default='rme', type=str, help='the model to run: rme, cofactor')
parser.add_argument('--n_factors', default=40, type=int, help='number of hidden factors for user/item representation')
parser.add_argument('--reg', default=1.0, type=float, help='regularization for user and item latent factors (alpha, beta)')
parser.add_argument('--reg_embed', default=1.0, type=float, help='regularization for user and item context latent factors (gamma, delta, theta)')
parser.add_argument('--dataset', default="ml10m", type=str, help='dataset')
parser.add_argument('--neg_item_inference', default=0, type=int, help='if there is no available disliked items, set this to 1 to infer '
                                                                      'negative items for users using our user-oriented EM like algorithm')
parser.add_argument('--neg_sample_ratio', default=0.2, type=float, help='negative sample ratio per user. If a user consumed 10 items, and this'
                                                                        'neg_sample_ratio = 0.2 --> randomly sample 2 negative items for the user')

args = parser.parse_args()


DATA_DIR = os.path.join(args.data_path, args.dataset)
gc.DATA_DIR = DATA_DIR
gc.SAVED_MODLE_DIR = args.saved_model_path
gc.PRED_DIR = os.path.join(DATA_DIR, 'prediction-temp')
SHIFTED_K_VALUE = args.s
NEGATIVE_SAMPLE_RATIO = args.neg_sample_ratio
save_dir = os.path.join(DATA_DIR, 'model_tmp_res')
n_components = args.n_factors
lam = args.reg
lam_emb = args.reg_embed

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
n_items = len(unique_movieId)
print n_users, n_items

def load_data(csv_file, shape=(n_users, n_items)):
    tp = pd.read_csv(csv_file)
    rows, cols = np.array(tp['userId']), np.array(tp['movieId'])
    seq = np.concatenate((rows[:, None], cols[:, None], np.ones((rows.size, 1), dtype='int')), axis=1)
    data = sparse.csr_matrix((np.ones_like(rows), (rows, cols)), dtype=np.int16, shape=shape)
    return data, seq, tp

def get_row(M, i):
    # get the row i of sparse matrix:
    lo, hi = M.indptr[i], M.indptr[i + 1]
    return lo, hi, M.data[lo:hi], M.indices[lo:hi]


def convert_to_SPPMI_matrix(M, max_row, shifted_K=1):
    # if we sum the co-occurrence matrix by row wise or column wise --> we have an array that contain the #(i) values
    obj_counts = np.asarray(M.sum(axis=1)).ravel()
    total_obj_pairs = M.data.sum()
    M_sppmi = M.copy()
    for i in xrange(max_row):
        lo, hi, data, indices = get_row(M, i)
        M_sppmi.data[lo:hi] = np.log(data * total_obj_pairs / (obj_counts[i] * obj_counts[indices]))
    M_sppmi.data[M_sppmi.data < 0] = 0
    M_sppmi.eliminate_zeros()
    if shifted_K == 1:
        return M_sppmi
    else:
        M_sppmi.data -= np.log(shifted_K)
        M_sppmi.data[M_sppmi.data < 0] = 0
        M_sppmi.eliminate_zeros()
    return M_sppmi


if args.neg_item_inference:
    #initialize with WMF:
    import wmf
    import rec_eval
    from scipy import sparse
    import produce_negative_embedding as pne
    import glob
    import os

    def softmax(x):
        """Compute softmax values for each ranked list."""
        # We want the item with higher ranking score have lower prob to be withdrawn as negative instances#
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
    def compute_neg_prob(ranks):
        return softmax(np.negative(ranks))
    def _make_prediction(train_data, Et, Eb, user_idx, batch_users, mu=None,
                         vad_data=None):
        n_songs = train_data.shape[1]
        # exclude examples from training and validation (if any)
        item_idx = np.zeros((batch_users, n_songs), dtype=bool)
        item_idx[train_data[user_idx].nonzero()] = True
        if vad_data is not None:
            item_idx[vad_data[user_idx].nonzero()] = True
        X_pred = Et[user_idx].dot(Eb)
        if mu is not None:
            if isinstance(mu, np.ndarray):
                assert mu.size == n_songs  # mu_i
                X_pred *= mu
            elif isinstance(mu, dict):  # func(mu_ui)
                params, func = mu['params'], mu['func']
                args = [params[0][user_idx], params[1]]
                if len(params) > 2:  # for bias term in document or length-scale
                    args += [params[2][user_idx]]
                if not callable(func):
                    raise TypeError("expecting a callable function")
                X_pred *= func(*args)
            else:
                raise ValueError("unsupported mu type")
        X_pred[item_idx] = np.inf
        return X_pred

    def gen_neg_instances(train_data, U, VT, user_idx, neg_ratio = 1.0, iter = 0):
        print 'Job start... %d to %d'%(user_idx.start, user_idx.stop)
        #if user_idx.start != 99000: return
        batch_users = user_idx.stop - user_idx.start
        X_pred = _make_prediction(train_data, U, VT, user_idx, batch_users, vad_data=vad_data)

        rows = []
        cols = []
        total_lost = 0
        for idx, uid in enumerate(range(user_idx.start, user_idx.stop)):
            num_pos = train_data[uid].count_nonzero()
            num_neg = int(num_pos * neg_ratio)
            if num_neg <= 0: continue
            ranks = X_pred[idx]
            neg_withdrawn_prob = compute_neg_prob(ranks)
            # print (neg_withdrawn_prob)
            neg_instances = list(set(np.random.choice(range(n_items), num_neg, p = neg_withdrawn_prob)))
            #rows = rows + len(neg_instances)*[uid]
            #uid_dup = np.empty(len(neg_instances))
            #uid_dup.fill(uid)
            if uid < 0: print 'error with %d to %d'%(user_idx.start, user_idx.stop)
            #rows = rows + uid_dup
            rows = np.append(rows, np.full( len(neg_instances), uid )  )
            cols = np.append(cols, neg_instances)
        # print 'check for neg values: ', np.sum(rows <0)
        # print 'check for neg values: ', np.sum(cols <0)
        if  len(rows) > 0:
            path = os.path.join(DATA_DIR, 'sub_dataframe_iter_%d_idxstart_%d.csv' % (iter, user_idx.start))
            assert len(rows) == len(cols)
            with open(path, 'w') as writer:
                for i in range(len(rows)): writer.write(str(rows[i]) + "," + str(cols[i]) + '\n')
                writer.flush()
            #df = pd.DataFrame({'uid':rows, 'sid':cols}, columns=["uid", "sid"], dtype=np.int16)
            #df.to_csv(path, sep=",",header=False, index = False)
        # return df
    U, V = None, None



    vad_data, vad_raw, vad_df = load_data(os.path.join(DATA_DIR, 'validation.csv'))
    train_data, train_raw, train_df = load_data(os.path.join(DATA_DIR, 'train.csv'))
    test_data, test_raw, test_df = load_data(os.path.join(DATA_DIR, 'test.csv'))
    U, V = wmf.decompose(train_data, vad_data, num_factors= n_components)
    VT = V.T
    iter, max_iter = 0, 10

    #load postivie information
    X = text_utils.load_pickle(os.path.join(DATA_DIR, 'item_item_cooc.dat'))
    Y = text_utils.load_pickle(os.path.join(DATA_DIR, 'user_user_cooc.dat'))
    X_sppmi = convert_to_SPPMI_matrix(X, max_row=n_items, shifted_K=SHIFTED_K_VALUE)
    Y_sppmi = convert_to_SPPMI_matrix(Y, max_row=n_users, shifted_K=SHIFTED_K_VALUE)

    best_ndcg100 = 0.0
    best_iter = 1
    early_stopping = False
    while (iter < max_iter and not early_stopping):
        ################ Expectation step: ######################
        user_slices = rec_eval.user_idx_generator(n_users, batch_users=5000)
        print 'GENERATING NEGATIVE INSTANCES ...'
        t1 = time.time()
        df = Parallel(n_jobs=16)(delayed(gen_neg_instances)(train_data, U, VT, user_idx, neg_ratio = NEGATIVE_SAMPLE_RATIO, iter = iter)
                                      for user_idx in user_slices)
        t2 = time.time()
        print 'Time : %d seconds' % (t2 - t1)


        print 'merging to one file ...'
        t1 = time.time()
        neg_file_out = os.path.join(DATA_DIR, 'train_neg_iter_%d.csv' % (iter))
        with open(neg_file_out, 'w') as writer:
            writer.write('userId,movieId\n')
        # os.system("echo uid,sid >> " + neg_file_out)
        for f in glob.glob(os.path.join(DATA_DIR, 'sub_dataframe_iter*')):
            os.system("cat " + f + " >> " + neg_file_out)
                # with open(f, 'rb') as reader:
                #
                #     writer.write(reader.readline())
            # writer.flush()
        #clean
        for f in glob.glob(os.path.join(DATA_DIR, 'sub_dataframe_iter*')):
            os.remove(f)

        t2 = time.time()
        print 'Time : %d seconds' % (t2 - t1)
        # neg_train_df = pd.concat(df)
        # neg_train_df.to_csv(neg_file_out, index = False)
        #########################################################

        ################ maximization step:######################
        print 'GENERATING NEGATIVE EMBEDDINGS ...'
        t1 = time.time()
        train_neg_data, _, train_neg_df = load_data(neg_file_out, shape=(n_users, n_items))
        #build the negative info:
        X_neg, _ = pne.produce_neg_embeddings(DATA_DIR, train_neg_data, n_users, n_items, iter = iter)
        X_neg_sppmi = convert_to_SPPMI_matrix(X_neg, max_row=n_items, shifted_K=SHIFTED_K_VALUE)
        Y_neg_sppmi = None
        t2 = time.time()
        print 'Time : %d seconds' % (t2 - t1)

        # build the model
        print 'build the model...'
        t1 = time.time()
        runner = ModelRunner(train_data, vad_data, None, X_sppmi, X_neg_sppmi, Y_sppmi, None, save_dir=save_dir)
        U, V, ndcg100 = runner.run("rme", n_jobs = 1,
                                         lam=lam, lam_emb=lam_emb, n_components = n_components, ret_params_only = 1)
        t2 = time.time()
        print 'Time : %d seconds' % (t2 - t1)
        print '*************************************ITER %d ******************************************' % iter
        print 'NDCG@100 at this iter:',ndcg100
        #
        if best_ndcg100 < ndcg100:
            best_iter = iter
            best_ndcg100 = ndcg100
        else:
            early_stopping = True
        iter += 1
        #########################################################
    print 'Max NDCG@100: %f , at iter: %d'%(best_ndcg100, best_iter)
    best_train_neg_file = os.path.join(DATA_DIR, 'train_neg_iter_%d.csv' % (best_iter))
    best_train_neg_file_newname = os.path.join(DATA_DIR, 'train_neg.csv')
    best_train_emb_file = os.path.join(DATA_DIR, 'negative_item_item_cooc_iter%d.dat' % (best_iter))
    best_train_emb_file_newname = os.path.join(DATA_DIR, 'negative_item_item_cooc.dat')
    print 'renaming from %s to %s'%(best_train_neg_file, best_train_neg_file_newname)
    os.rename(best_train_neg_file, best_train_neg_file_newname)
    print 'renaming from %s to %s' % (best_train_emb_file, best_train_emb_file_newname)
    os.rename(best_train_emb_file, best_train_emb_file_newname)
    #cleaning
    for i in range(max_iter):
        if i == best_iter: continue
        if early_stopping and (i > best_iter + 1): break
        del_file = os.path.join(DATA_DIR, 'train_neg_iter_%d.csv' % ( i))
        os.remove(del_file)
        del_file = os.path.join(DATA_DIR, 'negative_item_item_cooc_iter%d.dat' % (i))
        os.remove(del_file)




LOAD_NEGATIVE_MATRIX = True
if args.model.lower() != 'rme':
    LOAD_NEGATIVE_MATRIX = False
recalls = np.zeros(5, dtype=np.float32) #store results of topk recommendation in range [5, 10, 20, 50, 100]
ndcgs = np.zeros(5, dtype=np.float32)
maps = np.zeros(5, dtype=np.float32)
print '*************************************lam =  %.3f ******************************************' % lam
print '*************************************lam embedding =  %.3f ******************************************' % lam_emb

# train_data, train_raw, train_df = load_data(os.path.join(DATA_DIR, 'train_fold%d.csv'%FOLD))
vad_data, vad_raw, vad_df = load_data(os.path.join(DATA_DIR, 'validation.csv'))
test_data, test_raw, test_df = load_data(os.path.join(DATA_DIR, 'test.csv'))
train_data, train_raw, train_df = load_data(os.path.join(DATA_DIR, 'train.csv' ))

print 'loading pro_pro_cooc.dat'
t1 = time.time()
X = text_utils.load_pickle(os.path.join(DATA_DIR, 'item_item_cooc.dat'))
t2 = time.time()
print '[INFO]: sparse matrix size of item item co-occurrence matrix: %d mb\n' % (
    (X.data.nbytes + X.indices.nbytes + X.indptr.nbytes) / (1024 * 1024))
print 'Time : %d seconds' % (t2 - t1)

print 'loading user_user_cooc.dat'
t1 = time.time()
Y = text_utils.load_pickle(os.path.join(DATA_DIR, 'user_user_cooc.dat'))
t2 = time.time()
print '[INFO]: sparse matrix size of user user co-occurrence matrix: %d mb\n' % (
    (Y.data.nbytes + Y.indices.nbytes + Y.indptr.nbytes) / (1024 * 1024))
print 'Time : %d seconds' % (t2 - t1)
################# LOADING NEGATIVE CO-OCCURRENCE MATRIX ########################################

if LOAD_NEGATIVE_MATRIX:
    print 'test loading negative_pro_pro_cooc.dat'
    t1 = time.time()
    X_neg = text_utils.load_pickle(os.path.join(DATA_DIR, 'negative_item_item_cooc.dat'))
    t2 = time.time()
    print '[INFO]: sparse matrix size of negative item item co-occurrence matrix: %d mb\n' % (
        (X_neg.data.nbytes + X_neg.indices.nbytes + X_neg.indptr.nbytes) / (1024 * 1024))
    print 'Time : %d seconds' % (t2 - t1)


################################################################################################
########## converting CO-OCCURRENCE MATRIX INTO Shifted Positive Pointwise Mutual Information (SPPMI) matrix ###########
####### We already know the user-user co-occurrence matrix Y and item-item co-occurrence matrix X

print 'converting co-occurrence matrix into sppmi matrix'
t1 = time.time()
X_sppmi = convert_to_SPPMI_matrix(X, max_row=n_items, shifted_K=SHIFTED_K_VALUE)
Y_sppmi = convert_to_SPPMI_matrix(Y, max_row=n_users, shifted_K=SHIFTED_K_VALUE)
t2 = time.time()
print 'Time : %d seconds' % (t2 - t1)
# if DEBUG_MODE:
#     print 'item sppmi matrix'
#     print X_sppmi
#     print 'user sppmi matrix'
#     print Y_sppmi

############### Negative SPPMI matrix ##########################
X_neg_sppmi = None
Y_neg_sppmi = None
if LOAD_NEGATIVE_MATRIX:
    print 'converting negative co-occurrence matrix into sppmi matrix'
    t1 = time.time()
    X_neg_sppmi = convert_to_SPPMI_matrix(X_neg, max_row=n_items, shifted_K=SHIFTED_K_VALUE)
    t2 = time.time()
    print 'Time : %d seconds' % (t2 - t1)
################################################################


######## Finally, we have train_data, vad_data, test_data,
# X_sppmi: item item Shifted Positive Pointwise Mutual Information matrix
# Y_sppmi: user-user       Shifted Positive Pointwise Mutual Information matrix


print 'Training data', train_data.shape
print 'Validation data', vad_data.shape
print 'Testing data', test_data.shape

n_jobs = 1  # default value
model_type = 'model2'  # default value
if os.path.exists(save_dir):
    #clearning folder
    lst = glob.glob(os.path.join(save_dir, '*.*'))
    for f in lst:
        os.remove(f)
else:
    os.mkdir(save_dir)


runner = ModelRunner(train_data, vad_data, test_data, X_sppmi, X_neg_sppmi, Y_sppmi, Y_neg_sppmi,
                       save_dir=save_dir)

start = time.time()
if args.model == 'wmf':
    (recalls, ndcgs, maps) = runner.run("wmf", n_jobs=n_jobs, lam=lam,
                                                         saved_model = True,
                                                         n_components = n_components)
if args.model == 'cofactor':
    (recalls, ndcgs, maps) = runner.run("cofactor", n_jobs=n_jobs,
                                                        lam=lam,
                                                         saved_model=True,
                                                         n_components=n_components)
if args.model == 'rme':
    (recalls, ndcgs, maps) = runner.run("rme", n_jobs=n_jobs,lam=lam, lam_emb = lam_emb,
                                                         saved_model=True,
                                                         n_components=n_components)
end = time.time()
print ('total running time: %d seconds'%(end-start))
for idx, topk in enumerate([5, 10, 20, 50, 100]):
    print 'top-%d results: recall@%d = %.4f, ndcg@%d = %.4f, map@%d = %.4f'%(topk,
                                                                                  topk, recalls[idx],
                                                                                  topk, ndcgs[idx],
                                                                                  topk, maps[idx])


