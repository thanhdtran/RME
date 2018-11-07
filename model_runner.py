
import os
import numpy as np
import rec_eval as rec_eval
import global_constants as constants
from parallel_rme import ParallelRME
import cofactor as cofactor
import wmf
import glob
import sys
class ModelRunner:

    def __init__(self, train_data, vad_data, test_data,
                 X_sppmi, X_neg_sppmi, Y_sppmi, Y_neg_sppmi,
                 save_dir ):

        self.save_dir = save_dir

        self.train_data = train_data
        self.test_data = test_data
        self.vad_data = vad_data
        self.X_sppmi = X_sppmi
        self.Y_sppmi = Y_sppmi
        self.X_neg_sppmi = X_neg_sppmi
        self.Y_neg_sppmi = Y_neg_sppmi

        self.n_components = 100
        self.lam, self.lam_emb = 1e-1, 1e-1
    def clean_savedir(self):
        print 'cleaning folder'
        lst = glob.glob(os.path.join(self.save_dir, '*.npz'))
        for f in lst:
            os.remove(f)
    def cal_ndcg(self, U, V, K = 100):
        return rec_eval.parallel_normalized_dcg_at_k(self.train_data, self.vad_data, U, V, k=K,
                                                              vad_data=None, n_jobs=16, clear_invalid=False)
    def eval(self, U, V, ranges = [5,10,20,50,100]):
        recall_all = np.zeros(5, dtype=np.float32)
        ndcg_all = np.zeros(5, dtype=np.float32)  
        map_all = np.zeros(5, dtype=np.float32)
        PRED_DIR = os.path.join(constants.DATA_DIR, 'prediction-temp')
        if not os.path.exists(PRED_DIR): os.mkdir(PRED_DIR)
        else:
            for f in glob.glob(os.path.join(PRED_DIR, '*.npz')):
                os.remove(f)
        print 'n_components = %d, lam = %.4f, lam_emb = %.4f'%(self.n_components, self.lam, self.lam_emb)
        for index, K in enumerate(ranges):
            recall_at_K = rec_eval.parallel_recall_at_k(self.train_data, self.test_data, U, V, k=K,
                                                        vad_data=self.vad_data, n_jobs=16, clear_invalid=False, cache=True)
            print 'Test Recall@%d: \t %.4f' % (K, recall_at_K)
            ndcg_at_K = rec_eval.parallel_normalized_dcg_at_k(self.train_data, self.test_data, U, V, k=K,
                                                              vad_data=self.vad_data, n_jobs=16, clear_invalid=False, cache=True)
            print 'Test NDCG@%d: \t %.4f' % (K, ndcg_at_K)
            map_at_K = rec_eval.parallel_map_at_k(self.train_data, self.test_data, U, V, k=K,
                                                  vad_data=self.vad_data, n_jobs=16, clear_invalid=False, cache=True)
            print 'Test MAP@%d: \t %.4f' % (K, map_at_K)
            #if K == 100:
            #    recall100 = recall_at_K
            #    ndcg100 = ndcg_at_K
            #    map100 = map_at_K
            recall_all[index] = recall_at_K
            ndcg_all[index] = ndcg_at_K
            map_all[index] = map_at_K
        #clean
        for f in glob.glob(os.path.join(PRED_DIR, '*.npz')):
            # print 'removing ', f
            os.remove(f)
        return (recall_all, ndcg_all, map_all)

    def run(self, type, n_jobs = 16, n_components = 100, max_iter = 50, vad_K = 100, **kwargs):
        saved_model = kwargs.get('saved_model', False)
        if saved_model:
            MODELS_DIR = 'MODELS'
            if not os.path.exists(MODELS_DIR): os.mkdir(MODELS_DIR)
        lam = kwargs.get('lam', 1e-1)
        lam_alpha = lam_beta = lam
        lam_emb = kwargs.get('lam_emb', lam)
        lam_theta = lam_gamma = lam_gamma_p = lam_gamma_n = lam_theta_p = lam_emb
        c0 = 1.
        c1 = 10.


        self.n_components, self.lam, self.lam_emb = n_components, lam, lam_emb


        print '*************************************lam =  %.3f ******************************************' % lam
        print '*************************************lam embedding =  %.3f ******************************************' % lam_emb
        if type == 'wmf':
            U, V = wmf.decompose(self.train_data, self.vad_data, num_factors=n_components, lam=lam)
            (recall_all, ndcg_all, map_all) = self.eval(U, V)
            if saved_model:
                model_out_name = os.path.join(constants.SAVED_MODLE_DIR, 'WMF_K%d_lambda%.4f.npz' % (n_components, lam))
                np.savez(model_out_name, U=U, V=V)

        elif type == 'cofactor':
            print 'cofactor model'
            print self.save_dir
            self.clean_savedir()
            CoFacto = cofactor.CoFacto(
                n_components=n_components, max_iter=max_iter, batch_size=1000, init_std=0.01, dtype=np.float32,
                n_jobs=n_jobs,
                random_state=98765, save_params=True, save_dir=self.save_dir, early_stopping=True, verbose=True,
                lambda_alpha=lam_alpha, lambda_theta=lam_theta, lambda_beta=lam_beta, lambda_gamma=lam_gamma, c0=c0,
                c1=c1)
            CoFacto.fit(self.train_data, self.X_sppmi, vad_data=self.vad_data, batch_users=3000, k=vad_K,
                      clear_invalid=False, n_jobs = 16)
            self.test_data.data = np.ones_like(self.test_data.data)
            n_params = len(glob.glob(os.path.join(self.save_dir, '*.npz')))
            params = np.load(os.path.join(self.save_dir, 'CoFacto_K%d_iter%d.npz' % (n_components, n_params - 1)))
            U, V = params['U'], params['V']
            (recall_all, ndcg_all, map_all) = self.eval(U, V)
            recall100, ndcg100, map100 = recall_all[-1], ndcg_all[-1], map_all[-1]
            if saved_model:
                model_out_name = os.path.join(constants.SAVED_MODLE_DIR, 'Cofactor_K%d_lambda%.4f.npz' % (n_components, lam))
                np.savez(model_out_name, U=U, V=V)


        elif type == 'rme':
            ret_params_only = bool(kwargs.get("ret_params_only", False))
            print 'positive and negative project embedding + positive user embedding'
            mu_p_p = float(kwargs.get('mu_p_p', 1.0)) #weight to indicate importance of liked item embeddings
            mu_p_n = float(kwargs.get('mu_p_n', 1.0)) #weight to indicate importance of disliked item embeddings
            mu_u_p = float(kwargs.get('mu_u_p', 1.0)) #weight to indicate importance of user embeddings


            print 'mu_u_p = %.1f, mu_p_p = %.1f, mu_p_n = %.1f' % (mu_u_p, mu_p_p, mu_p_n)
            print self.save_dir
            self.clean_savedir()

            RME = ParallelRME(mu_u_p=mu_u_p, mu_p_p=mu_p_p, mu_p_n=mu_p_n,
                                 n_components=n_components, max_iter=max_iter, batch_size=3000, init_std=0.01, dtype=np.float32, n_jobs=n_jobs,
                                 random_state=98765, save_params=True, save_dir=self.save_dir, early_stopping=True, verbose=True,
                                 lambda_alpha=lam_alpha, lambda_theta_p=lam_theta_p,
                                 lambda_beta=lam_beta, lambda_gamma_p=lam_gamma_p, lambda_gamma_n=lam_gamma_n,
                                 c0=c0, c1=c1)
            RME.fit(self.train_data, self.X_sppmi, self.X_neg_sppmi, self.Y_sppmi,
                      vad_data=self.vad_data, batch_users=3000, k=vad_K, clear_invalid=False, n_jobs = 15)


            n_params = len(glob.glob(os.path.join(self.save_dir, '*.npz')))
            params = np.load(os.path.join(self.save_dir, 'RME_K%d_iter%d.npz' % (n_components, n_params - 1)))
            U, V = params['U'], params['V']
            if (ret_params_only):
                return (U, V, self.cal_ndcg(U,V,K=vad_K))
            self.test_data.data = np.ones_like(self.test_data.data)
            (recall_all, ndcg_all, map_all) = self.eval(U, V)
            recall100, ndcg100, map100 = recall_all[-1], ndcg_all[-1], map_all[-1]
            if saved_model:
                model_out_name = os.path.join(constants.SAVED_MODLE_DIR, 'RME_K%d_lambda%.4f.npz' % (n_components, lam))
                np.savez(model_out_name, U=U, V=V)

        else:
            print 'Please select model from: rme, cofactor, wmf'
            sys.exit(1)

        U, V = None, None
        return (recall_all, ndcg_all, map_all)



