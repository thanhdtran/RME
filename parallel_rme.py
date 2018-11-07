import os
import sys
import time
import numpy as np
import MultiProcessParallelSolver as mpps
from sklearn.base import BaseEstimator, TransformerMixin
import rec_eval


class ParallelRME(BaseEstimator, TransformerMixin):
    def __init__(self, mu_u_p = 1, mu_p_p = 1, mu_p_n = 1, n_components=100, max_iter=10, batch_size=1000,
                 init_std=0.01, dtype='float32', n_jobs=15, random_state=None,
                 save_params=False, save_dir='.', early_stopping=False,
                 verbose=False, **kwargs):
        self.mu_u_p = mu_u_p
        self.mu_p_p = mu_p_p
        self.mu_p_n = mu_p_n
        self.n_components = n_components
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.init_std = init_std
        self.dtype = dtype
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.save_params = save_params
        self.save_dir = save_dir
        self.early_stopping = early_stopping
        self.verbose = verbose

        if type(self.random_state) is int:
            np.random.seed(self.random_state)
        elif self.random_state is not None:
            np.random.setstate(self.random_state)

        self._parse_kwargs(**kwargs)

    def _parse_kwargs(self, **kwargs):
        ''' Model hyperparameters
        Parameters
        ---------
        lambda_alpha, lambda_beta, lambda_gamma: float
            Regularization parameter for user (lambda_alpha), item factors (
            lambda_beta), and context factors (lambda_gamma).
        c0, c1: float
            Confidence for 0 and 1 in Hu et al., c0 must be less than c1
        '''
        self.lam_alpha = float(kwargs.get('lambda_alpha', 1e-1))
        self.lam_beta = float(kwargs.get('lambda_beta', 1e-1))
        self.lam_theta_p = float(kwargs.get('lambda_gamma_p', 1e-1))
        self.lam_gamma_p = float(kwargs.get('lambda_gamma_p', 1e-1))
        self.lam_gamma_n = float(kwargs.get('lambda_gamma_n', 1e-5))
        self.c0 = float(kwargs.get('c0', 0.1))
        self.c1 = float(kwargs.get('c1', 2.0))
    
        print ('c0 : %.2f, c1: %.2f'%(self.c0, self.c1))
        print ('alpha: %.5f, beta: %.5f'%(self.lam_alpha, self.lam_beta))
        print ('theta_p: %.5f, gamma_p: %.5f, gamma_n: %.5f'%(self.lam_theta_p, self.lam_gamma_p, self.lam_gamma_n))
        assert self.c0 < self.c1, "c0 must be smaller than c1"

    def _init_params(self, n_users, n_projects):
        ''' Initialize all the latent factors and biases '''
        self.alpha = self.init_std * np.random.randn(n_users, self.n_components).astype(self.dtype)
        self.beta = self.init_std *  np.random.randn(n_projects, self.n_components).astype(self.dtype)
        self.theta_p = self.init_std * np.random.randn(n_users, self.n_components).astype(self.dtype)
        self.gamma_p = self.init_std * np.random.randn(n_projects, self.n_components).astype(self.dtype)
        self.bias_b_p = np.zeros(n_users, dtype=self.dtype)
        self.bias_c_p = np.zeros(n_users, dtype=self.dtype)
        self.bias_d_p = np.zeros(n_projects, dtype=self.dtype)
        self.bias_e_p = np.zeros(n_projects, dtype=self.dtype)
        # global bias
        self.global_x_p = 0.0
        self.global_y_p = 0.0  # intercept of second factorization for user-user

        self.gamma_n = self.init_std * np.random.randn(n_projects, self.n_components).astype(self.dtype)
        self.bias_d_n = np.zeros(n_projects, dtype=self.dtype)
        self.bias_e_n = np.zeros(n_projects, dtype=self.dtype)
        # global bias
        self.global_x_n = 0.0


    def fit(self, M, XP = None, XN = None, YP = None,
            FXP=None, FXN = None, FYP = None, vad_data=None, **kwargs):
        n_users, n_projects = M.shape
        assert XP.shape == (n_projects, n_projects)
        assert XN.shape == (n_projects, n_projects)
        assert YP.shape == (n_users, n_users)


        self._init_params(n_users, n_projects)
        self._update(M, XP, XN, YP,  FXP, FXN, FYP, vad_data, **kwargs)
        return self

    def transform(self, M):
        pass

    def _update(self, M, XP, XN, YP,  FXP, FXN, FYP, vad_data, **kwargs):
        '''Model training and evaluation on validation set'''
        MT = M.T.tocsr()  # pre-compute this
        XPT, XNT, FXPT, FXNT = None, None, None, None
        YPT, FYPT = None, None
        if XP != None:
            XPT = XP.T
        if XN != None:
            XNT = XN.T
        if FXP != None:
            FXPT = FXP.T
        if FXN != None:
            FXNT = FXN.T
        if YP != None:
            YPT = YP.T
        if FYP != None:
            FYPT = FYP.T

        self.vad_ndcg = -np.inf
        for i in xrange(self.max_iter):
            if self.verbose:
                print('ITERATION #%d' % i)
            self._update_factors(M, MT, XP, XPT, XN, XNT, YP, YPT,
                                 FXP, FXPT, FXN, FXNT, FYP, FYPT)
            self._update_biases(XP, XPT, XN, XNT, YP, YPT,
                                FXP, FXPT, FXN, FXNT, FYP, FYPT)
            if vad_data is not None:
                vad_ndcg = self._validate(M, vad_data, **kwargs)
                if self.early_stopping and self.vad_ndcg > vad_ndcg:
                    break  # we will not save the parameter for this iteration
                self.vad_ndcg = vad_ndcg
            if self.save_params:
                self._save_params(i)
        pass

    def _update_factors(self, M, MT, XP, XPT, XN, XNT, YP, YPT,
                        FXP, FXPT, FXN, FXNT, FYP, FYPT, ):
        if self.verbose:
            start_t = _writeline_and_time('\tUpdating user factors...')

        self.alpha = update_alpha(self.beta, self.theta_p,
                                  self.bias_b_p, self.bias_c_p, self.global_y_p,
                                  M, YP, FYP, self.c0, self.c1, self.lam_alpha,
                                  n_jobs=self.n_jobs, batch_size=self.batch_size,
                                  mu_u_p=self.mu_u_p)
        # print('checking user factor isnan : %d'%(np.sum(np.isnan(self.alpha))))
        if self.verbose:
            print('\r\tUpdating user factors: time=%.2f'
                  % (time.time() - start_t))
            start_t = _writeline_and_time('\tUpdating project factors...')
        self.beta = update_beta(self.alpha, self.gamma_p, self.gamma_n,
                                self.bias_d_p, self.bias_e_p, self.global_x_p,
                                self.bias_d_n, self.bias_e_n, self.global_x_n,
                                MT, XP, FXP, XN, FXN, self.c0, self.c1, self.lam_beta,
                                self.n_jobs, batch_size=self.batch_size,
                                mu_p_p = self.mu_p_p, mu_p_n = self.mu_p_n)
        # print('checking project factor isnan : %d' % (np.sum(np.isnan(self.beta))))
        if self.verbose:
            print('\r\tUpdating project factors: time=%.2f'
                  % (time.time() - start_t))
            start_t = _writeline_and_time('\tUpdating liked project embedding factors...')
        self.gamma_p = update_embedding_factor(self.beta,
                                             self.bias_d_p, self.bias_e_p, self.global_x_p,
                                             XPT, FXPT, self.lam_gamma_p,
                                             self.n_jobs,
                                             batch_size=self.batch_size,
                                             mu_p=self.mu_p_p)
        # print('checking gamma_p isnan : %d' % (np.sum(np.isnan(self.gamma_p))))
        if self.verbose:
            print('\r\tUpdating liked project embedding factors: time=%.2f'
                  % (time.time() - start_t))
            start_t = _writeline_and_time('\tUpdating disliked project embedding factors...')
        self.gamma_n = update_embedding_factor(self.beta,
                                             self.bias_d_n, self.bias_e_n, self.global_x_n,
                                             XNT, FXNT, self.lam_gamma_n,
                                             self.n_jobs,
                                             batch_size=self.batch_size,
                                             mu_p=self.mu_p_n)

        if self.verbose:
            print('\r\tUpdating disliked project embedding factors: time=%.2f'
                  % (time.time() - start_t))
            start_t = _writeline_and_time('\tUpdating user embedding factors...')
        self.theta_p = update_embedding_factor(self.alpha,
                                             self.bias_b_p, self.bias_c_p, self.global_y_p,
                                             YPT, FYPT, self.lam_theta_p,
                                             self.n_jobs,
                                             batch_size=self.batch_size,
                                             mu_p=self.mu_u_p)
        if self.verbose:
            print('\r\tUpdating user embedding factors: time=%.2f'
                  % (time.time() - start_t))



        pass

    def _update_biases(self, XP, XPT, XN, XNT, YP, YPT,
                       FXP, FXPT, FXN, FXNT, FYP, FYPT):
        if self.verbose:
            start_t = _writeline_and_time('\tUpdating bias terms...')

        self.bias_d_p = update_bias(self.beta, self.gamma_p,
                                  self.bias_e_p, self.global_x_p, XP, FXP,
                                  self.n_jobs, batch_size=self.batch_size,
                                  mu = self.mu_p_p)
        self.bias_e_p = update_bias(self.gamma_p, self.beta,
                                  self.bias_d_p, self.global_x_p, XP, FXP,
                                  self.n_jobs, batch_size=self.batch_size,
                                  mu=self.mu_p_p)


        self.global_x_p = update_global(self.beta, self.gamma_p,
                                  self.bias_d_p, self.bias_e_p, XP, FXP,
                                  self.n_jobs, batch_size=self.batch_size,
                                  mu=self.mu_p_p)


        self.bias_d_n = update_bias(self.beta, self.gamma_n,
                                    self.bias_e_n, self.global_x_n, XN, FXN,
                                    self.n_jobs, batch_size=self.batch_size,
                                    mu=self.mu_p_n)

        self.bias_e_n = update_bias(self.gamma_n, self.beta,
                                    self.bias_d_n, self.global_x_n, XN, FXN,
                                    self.n_jobs, batch_size=self.batch_size,
                                    mu=self.mu_p_n)

        self.global_x_n = update_global(self.beta, self.gamma_n,
                                        self.bias_d_n, self.bias_e_n, XN, FXN,
                                        self.n_jobs, batch_size=self.batch_size,
                                        mu=self.mu_p_n)


        self.bias_b_p = update_bias(self.alpha, self.theta_p,
                                    self.bias_c_p, self.global_y_p, YP, FYP,
                                    self.n_jobs, batch_size=self.batch_size,
                                    mu=self.mu_u_p)

        self.bias_c_p = update_bias(self.theta_p, self.alpha,
                                    self.bias_b_p, self.global_y_p, YP, FYP,
                                    self.n_jobs, batch_size=self.batch_size,
                                    mu=self.mu_u_p)
        self.global_y_p = update_global(self.alpha, self.theta_p,
                                        self.bias_b_p, self.bias_c_p, YP, FYP,
                                        self.n_jobs, batch_size=self.batch_size,
                                        mu=self.mu_u_p)

        if self.verbose:
            print('\r\tUpdating bias terms: time=%.2f'
                  % (time.time() - start_t))
        pass

    def _validate(self, M, vad_data, **kwargs):
        vad_ndcg = rec_eval.parallel_normalized_dcg_at_k(M, vad_data,
                                                self.alpha,
                                                self.beta,
                                                **kwargs)
        if self.verbose:
            print('\tValidation NDCG@k: %.5f' % vad_ndcg)
        return vad_ndcg

    def _save_params(self, iter):
        '''Save the parameters'''
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        filename = 'RME_K%d_iter%d.npz' % (self.n_components, iter)
        np.savez(os.path.join(self.save_dir, filename), U=self.alpha,
                 V=self.beta)


# Utility functions #
def _writeline_and_time(s):
    sys.stdout.write(s)
    sys.stdout.flush()
    return time.time()


def get_row(Y, i):
    '''Given a scipy.sparse.csr_matrix Y, get the values and indices of the
    non-zero values in i_th row'''
    lo, hi = Y.indptr[i], Y.indptr[i + 1]
    return Y.data[lo:hi], Y.indices[lo:hi]


def update_alpha(beta, theta_p,
                 bias_b_p, bias_c_p, global_y_p,
                 M, YP, FYP, c0, c1, lam_alpha,
                 n_jobs = 8, batch_size=1000, mu_u_p=1):
    '''Update user latent factors'''
    m, n = M.shape  # m: number of users, n: number of items
    f = beta.shape[1]  # f: number of factors

    BTB = c0 * np.dot(beta.T, beta)  # precompute this
    BTBpR = BTB + lam_alpha * np.eye(f, dtype=beta.dtype)

    return mpps.UpdateUserFactorParallel(
        beta, theta_p=theta_p, theta_n=None,
        bias_b_p=bias_b_p, bias_c_p=bias_c_p, global_y_p=global_y_p,
        bias_b_n=None, bias_c_n=None, global_y_n=None,
        M=M, YP=YP, FYP=FYP, YN=None, FYN=None, BTBpR=BTBpR,
        c0=c0, c1=c1, f=f, mu_u_p=mu_u_p, mu_u_n=None,
        n_jobs=n_jobs, mode='positive'
    ).run()

def update_beta(alpha, gamma_p, gamma_n,
                bias_d_p, bias_e_p, global_x_p,
                bias_d_n, bias_e_n, global_x_n,
                MT, XP, FXP, XN, FXN,
                c0, c1, lam_beta,
                n_jobs, batch_size=1000, mu_p_p = 1, mu_p_n = 1):
    '''Update item latent factors/embeddings'''
    n, m = MT.shape  # m: number of users, n: number of projects
    f = alpha.shape[1]
    assert alpha.shape[0] == m
    assert gamma_p.shape == (n, f)
    assert gamma_n.shape == (n, f)

    TTT = c0 * np.dot(alpha.T, alpha)  # precompute this
    TTTpR = TTT + lam_beta * np.eye(f, dtype=alpha.dtype)

    return mpps.UpdateProjectFactorParallel(
        alpha = alpha, gamma_p = gamma_p, gamma_n=gamma_n,
        bias_d_p=bias_d_p, bias_e_p=bias_e_p, global_x_p = global_x_p,
        bias_d_n=bias_d_n, bias_e_n=bias_e_n, global_x_n = global_x_n,
        MT = MT, XP = XP, FXP = FXP, XN = XN, FXN = FXN,
        TTTpR = TTTpR, c0=c0, c1=c1, f=f, mu_p_p=mu_p_p, mu_p_n=mu_p_n,
        n_jobs=n_jobs, mode='hybrid'
    ).run()

def update_embedding_factor(beta, bias_d, bias_e, global_x, XT, FXT, lam_gamma,
                 n_jobs, batch_size=1000, mu_p = 1):
    '''Update context latent factors'''
    n, f = beta.shape  # n: number of items, f: number of factors


    return mpps.UpdateEmbeddingFactorParallel(
        main_factor = beta, bias_main=bias_d, bias_embedding=bias_e,
        intercept=global_x, XT=XT, FXT=FXT, f=f, lam_embedding=lam_gamma,
        mu=mu_p, n_jobs=n_jobs
    ).run()

def update_bias(beta, gamma, bias_e, global_x, X, FX, n_jobs = 8, batch_size=1000,
                        mu = 1):
    ''' Update the per-item (or context) bias term.
    '''
    n = beta.shape[0]


    return mpps.UpdateBiasParallel(
        main_factor=beta, embedding_factor=gamma,
        bias=bias_e, intercept=global_x, X=X, FX=FX, mu=mu,
        n_jobs=n_jobs
    ).run()

def update_global(beta, gamma, bias_d, bias_e, X, FX, n_jobs, batch_size=1000,
                  mu = 1):
    n = beta.shape[0]
    assert beta.shape == gamma.shape
    assert bias_d.shape == bias_e.shape

    return mpps.UpdateInterceptParallel(
        main_factor = beta, embedding_factor = gamma,
        bias_main = bias_d, bias_embedding = bias_e, X = X, FX = FX, mu=mu,
        n_jobs=n_jobs
    ).run()

