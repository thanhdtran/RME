import numpy as np
from multiprocessing import Queue
import multiprocessing
from numpy import linalg as LA

def get_row(Y, i):
    '''Given a scipy.sparse.csr_matrix Y, get the values and indices of the
    non-zero values in i_th row'''
    lo, hi = Y.indptr[i], Y.indptr[i + 1]
    return Y.data[lo:hi], Y.indices[lo:hi]

def UserFactorUpdateWorker(out_q, lo, hi, beta, theta_p, theta_n,
           bias_b_p, bias_c_p, global_y_p,
           bias_b_n, bias_c_n, global_y_n,
           M, YP, FYP, YN, FYN, BTBpR, c0, c1, f, mu_u_p, mu_u_n, mode):
    alpha_batch = np.zeros((hi - lo, f), dtype=beta.dtype)
    if mode == None:
        #update user factor without embedding
        for ui, u in enumerate(xrange(lo, hi)):
            m_u, idx_m_p = get_row(M, u)
            B_p = beta[idx_m_p]
            a = m_u.dot(c1 * B_p)
            A = BTBpR + B_p.T.dot((c1 - c0) * B_p)
            alpha_batch[ui] = LA.solve(A, a)

    elif mode == "positive":
        for ui, u in enumerate(xrange(lo, hi)):
            m_u, idx_m_p = get_row(M, u)
            B_p = beta[idx_m_p]

            y_u, idx_y_u = get_row(YP, u)
            T_j = theta_p[idx_y_u]

            rsd = y_u - bias_b_p[u] - bias_c_p[idx_y_u] - global_y_p

            if FYP is not None:  # FY is weighted matrix of Y
                f_u, _ = get_row(FYP, u)
                TTT = T_j.T.dot(T_j * f_u[:, np.newaxis])
                rsd *= f_u
            else:
                TTT = T_j.T.dot(T_j)

            TTT = mu_u_p * TTT
            a = m_u.dot(c1 * B_p) + np.dot(rsd, T_j)
            A = BTBpR + B_p.T.dot((c1 - c0) * B_p) + TTT
            alpha_batch[ui] = LA.solve(A, a)

    out_q.put([lo, hi, alpha_batch])

class UpdateUserFactorParallel:
    def __init__(self,
                 beta, theta_p = None, theta_n = None,
                 bias_b_p = None, bias_c_p = None, global_y_p = None,
                 bias_b_n = None, bias_c_n = None, global_y_n = None,
                 M = None, YP = None, FYP = None, YN = None, FYN = None, BTBpR = None,
                 c0 = 1.0, c1 = 20.0, f = 100, mu_u_p = 1, mu_u_n = 1,
                 n_jobs = 15,  mode=None):

        self.beta = beta
        self.theta_p = theta_p
        self.theta_n = theta_n
        self.bias_b_p = bias_b_p
        self.bias_c_p = bias_c_p
        self.global_y_p = global_y_p
        self.bias_b_n = bias_b_n
        self.bias_c_n = bias_c_n
        self.global_y_n = global_y_n
        self.M = M
        self.YP = YP
        self.Y = YP
        self.FYP = FYP
        self.YN = YN
        self.FYN = FYN
        self.BTBpR = BTBpR
        self.c0 = c0
        self.c1 = c1
        self.f = f #number of latent factors
        self.n_jobs = n_jobs
        self.mu_u_p = mu_u_p
        self.mu_u_n = mu_u_n
        self.mode = mode
        self.m = M.shape[0]  # m: number of users

    def run(self):
        out_q = Queue()
        alpha = np.zeros((self.m, self.f), dtype=self.beta.dtype)
        step = int(self.m/self.n_jobs)
        procs = []
        for i in range(0, self.n_jobs):
            lo = i*step
            hi = (i+1)*step
            if i == (self.n_jobs - 1):
                hi = self.m
            p = multiprocessing.Process(
                target=UserFactorUpdateWorker,
                args=(out_q, lo, hi, self.beta, self.theta_p, self.theta_n,
                      self.bias_b_p, self.bias_c_p, self.global_y_p,
                      self.bias_b_n, self.bias_c_n, self.global_y_n,
                      self.M, self.YP, self.FYP, self.YN, self.FYN,
                      self.BTBpR, self.c0, self.c1, self.f, self.mu_u_p, self.mu_u_n, self.mode,)
            )

            p.start()
            procs.append(p)

        for i in range(self.n_jobs):
            [lo, hi, new_alpha] = out_q.get()
            alpha[lo:hi] = new_alpha
        for p in procs:
            p.join()

        return alpha



class UpdateProjectFactorParallel:
    def __init__(self,
                 alpha, gamma_p = None, gamma_n = None,
                 bias_d_p = None, bias_e_p = None, global_x_p = None,
                 bias_d_n = None, bias_e_n = None, global_x_n = None,
                 MT = None, XP = None, FXP = None, XN = None, FXN = None, TTTpR = None,
                 c0 = None, c1 = None, f = None, mu_p_p = 1, mu_p_n = 1,
                 n_jobs = 15, mode=None):

        self.alpha = alpha
        self.gamma_p = gamma_p
        self.gamma_n = gamma_n
        self.bias_d_p = bias_d_p
        self.bias_e_p = bias_e_p
        self.global_x_p = global_x_p
        self.bias_d_n = bias_d_n
        self.bias_e_n = bias_e_n
        self.global_x_n = global_x_n
        self.MT = MT
        self.XP = XP
        self.FXP = FXP
        self.XN = XN
        self.FXN = FXN
        self.TTTpR = TTTpR
        self.c0 = c0
        self.c1 = c1
        self.f = f
        self.mu_p_p = mu_p_p
        self.mu_p_n = mu_p_n
        self.n_jobs = n_jobs
        self.mode = mode
        self.n = MT.shape[0]  # n: number of projects, m: number of users



    def run(self):
        out_q = Queue()
        beta = np.zeros((self.n, self.f), dtype=self.alpha.dtype)
        step = int(self.n / self.n_jobs)
        procs = []
        for i in range(0, self.n_jobs):
            lo = i * step
            hi = (i + 1) * step
            if i == (self.n_jobs - 1):
                hi = self.n
            p = multiprocessing.Process(
                target=ProjectFactorUpdateWorker,
                args=(out_q, lo, hi,
                      self.alpha, self.gamma_p, self.gamma_n,
                      self.bias_d_p, self.bias_e_p, self.global_x_p,
                      self.bias_d_n, self.bias_e_n, self.global_x_n,
                      self.MT, self.XP, self.FXP, self.XN, self.FXN,
                      self.TTTpR, self.c0, self.c1, self.f,
                      self.mu_p_p, self.mu_p_n, self.mode,)
            )
            p.start()
            procs.append(p)
        for i in range(self.n_jobs):
            [lo, hi, new_beta] = out_q.get()
            beta[lo:hi] = new_beta
        for p in procs:
            p.join()
        return beta


def ProjectFactorUpdateWorker(out_q, lo, hi,
                              alpha, gamma_p, gamma_n,
                              bias_d_p, bias_e_p, global_x_p,
                              bias_d_n, bias_e_n, global_x_n,
                              MT, XP, FXP, XN, FXN, TTTpR, c0, c1, f,
                              mu_p_p, mu_p_n, mode):
    beta_batch = np.zeros((hi - lo, f), dtype=alpha.dtype)
    if mode == None:

        for pi, p in enumerate(xrange(lo, hi)):
            m_u, idx_u = get_row(MT, p)
            A_u = alpha[idx_u]

            a = m_u.dot(c1*A_u)
            B = TTTpR + A_u.T.dot((c1 - c0) * A_u)
            beta_batch[pi] = LA.solve(B, a)

    elif mode == 'hybrid':
        #update project latent factor with pos and neg embedding
        for pi, p in enumerate(xrange(lo, hi)):
            m_u, idx_u = get_row(MT, p)
            A_u = alpha[idx_u]

            x_pj_p, idx_x_j_p = get_row(XP, p)
            G_i_p = gamma_p[idx_x_j_p]
            rsd_p = x_pj_p - bias_d_p[p] - bias_e_p[idx_x_j_p] - global_x_p
            if FXP is not None:
                f_i_p, _ = get_row(FXP, p)
                GTG_p = G_i_p.T.dot(G_i_p * f_i_p[:, np.newaxis])
                rsd_p *= f_i_p
            else:
                GTG_p = G_i_p.T.dot(G_i_p)

            x_pj_n, idx_x_j_n = get_row(XN, p)
            G_i_n = gamma_n[idx_x_j_n]
            rsd_n = x_pj_n - bias_d_n[p] - bias_e_n[idx_x_j_n] - global_x_n
            if FXN is not None:
                f_i_n, _ = get_row(FXN, p)
                GTG_n = G_i_n.T.dot(G_i_n * f_i_n[:, np.newaxis])
                rsd_n *= f_i_n
            else:
                GTG_n = G_i_n.T.dot(G_i_n)

            B = TTTpR + A_u.T.dot((c1 - c0) * A_u) + \
                c1* mu_p_p*GTG_p + c1*mu_p_n*GTG_n
            a = m_u.dot(c1 * A_u) + c1*mu_p_p * np.dot(rsd_p, G_i_p) + \
                                    c1*mu_p_n * np.dot(rsd_n, G_i_n)

            beta_batch[pi] = LA.solve(B, a)

    out_q.put([lo, hi, beta_batch])

class UpdateEmbeddingFactorParallel:
    # beta, bias_d, bias_e, global_x, XT, FXT, f, lam_gamma, mu_p
    def __init__(self,
                 main_factor, bias_main, bias_embedding, intercept, XT, FXT, f, lam_embedding, mu=1,
                 n_jobs = 15):


        self.main_factor = main_factor
        self.bias_main = bias_main
        self.bias_embedding = bias_embedding
        self.intercept = intercept
        self.XT = XT
        self.FXT = FXT
        self.f =  f
        self.lam_embedding = lam_embedding
        self.mu = mu
        self.n_jobs = n_jobs
        self.n = main_factor.shape[0]


    def run(self):
        out_q = Queue()
        embedding_factor = np.zeros((self.n, self.f), dtype=self.main_factor.dtype)
        step = int(self.n / self.n_jobs)
        procs = []
        for i in range(0, self.n_jobs):
            lo = i * step
            hi = (i + 1) * step
            if i == (self.n_jobs - 1):
                hi = self.n
            p = multiprocessing.Process(
                target=EmbeddingFactorUpdateWorker,
                args=(out_q, lo, hi,
                      self.main_factor, self.bias_main, self.bias_embedding, self.intercept,
                      self.XT, self.FXT, self.f, self.lam_embedding, self.mu,)
            )
            p.start()
            procs.append(p)
        for i in range(self.n_jobs):
            [lo, hi, new_embedding_factor] = out_q.get()
            embedding_factor[lo:hi] = new_embedding_factor
        for p in procs:
            p.join()
        return embedding_factor

def EmbeddingFactorUpdateWorker(out_q, lo, hi,
                                main_factor, bias_main, bias_embedding, intercept,
                                XT, FXT, f, lam_embedding, mu):

    embedding_batch = np.zeros((hi - lo, f), dtype=main_factor.dtype)
    for jb, j in enumerate(xrange(lo, hi)):
        x_jp, idx_p = get_row(XT, j)
        rsd = x_jp - bias_main[idx_p] - bias_embedding[j] - intercept
        B_j = main_factor[idx_p]
        if FXT is not None:
            f_j, _ = get_row(FXT, j)
            BTB = B_j.T.dot(B_j * f_j[:, np.newaxis])
            rsd *= f_j
        else:
            BTB = B_j.T.dot(B_j)

        B = BTB + lam_embedding * np.eye(f, dtype=main_factor.dtype)
        a = mu * np.dot(rsd, B_j)
        embedding_batch[jb] = LA.solve(B, a)
    out_q.put([lo, hi, embedding_batch])


class UpdateBiasParallel:
    #beta, gamma, bias_e, global_x, X, FX, mu
    def __init__(self,
                 main_factor, embedding_factor, bias, intercept, X, FX, mu=1,
                 n_jobs = 15):


        self.main_factor = main_factor
        self.embedding_factor = embedding_factor
        self.bias = bias
        self.intercept = intercept
        self.X = X
        self.FX = FX
        self.mu = mu
        self.n_jobs = n_jobs
        self.n = main_factor.shape[0]

    def run(self):
        out_q = Queue()
        bias_update = np.zeros(self.n, dtype=self.main_factor.dtype)
        step = int(self.n / self.n_jobs)
        procs = []
        for i in range(0, self.n_jobs):
            lo = i * step
            hi = (i + 1) * step
            if i == (self.n_jobs - 1):
                hi = self.n
            p = multiprocessing.Process(
                target=BiasUpdateWorker,
                args=(out_q, lo, hi,
                      self.main_factor, self.embedding_factor, self.bias, self.intercept,
                      self.X, self.FX, self.mu,
                      )
            )
            p.start()
            procs.append(p)
        for i in range(self.n_jobs):
            [lo, hi, new_bias] = out_q.get()
            bias_update[lo:hi] = new_bias
        for p in procs:
            p.join()
        return bias_update

def BiasUpdateWorker(out_q, lo, hi,
                     main_factor, embedding_factor, bias, intercept,
                     X, FX, mu):
    bias_batch = np.zeros(hi - lo, dtype=main_factor.dtype)
    if mu != 0:
        for ib, i in enumerate(xrange(lo, hi)):
            m_i, idx_i = get_row(X, i)
            m_i_hat = embedding_factor[idx_i].dot(main_factor[i]) + \
                      bias[idx_i] + intercept
            rsd = m_i - m_i_hat

            if FX is not None:
                f_i, _ = get_row(FX, i)
                rsd *= f_i

            if rsd.size > 0:
                bias_batch[ib] = mu * rsd.mean()
            else:
                bias_batch[ib] = 0.
    out_q.put([lo, hi, bias_batch])


class UpdateInterceptParallel:
    #beta, gamma, bias_d, bias_e, X, FX, mu
    def __init__(self,
                 main_factor, embedding_factor, bias_main, bias_embedding, X, FX, mu=1,
                 n_jobs = 15):


        self.main_factor = main_factor
        self.embedding_factor = embedding_factor
        self.bias_main = bias_main
        self.bias_embedding = bias_embedding
        self.X = X
        self.FX = FX
        self.mu = mu
        self.n_jobs = n_jobs
        self.n = main_factor.shape[0]

    def run(self):
        out_q = Queue()
        intercept = np.zeros(self.n_jobs, dtype=float)
        step = int(self.n / self.n_jobs)
        procs = []
        for i in range(0, self.n_jobs):
            lo = i * step
            hi = (i + 1) * step
            if i == (self.n_jobs - 1):
                hi = self.n
            p = multiprocessing.Process(
                target=InterceptUpdateWorker,
                args=(out_q, i, lo, hi,
                      self.main_factor, self.embedding_factor, self.bias_main, self.bias_embedding,
                      self.X, self.FX, self.mu,
                      )
            )
            p.start()
            procs.append(p)
        for i in range(self.n_jobs):
            [process_id, lo, hi, new_intercept] = out_q.get()
            intercept[process_id] = new_intercept
        for p in procs:
            p.join()
        return np.sum(intercept) / self.X.data.size

def InterceptUpdateWorker(out_q, process_id, lo, hi,
                          main_factor, embedding_factor, bias_main, bias_embedding,
                          X, FX, mu
                          ):
    res = 0.
    if mu != 0:
        for ib, i in enumerate(xrange(lo, hi)):
            m_i, idx_i = get_row(X, i)
            m_i_hat = embedding_factor[idx_i].dot(main_factor[i]) + \
                      bias_main[i] + bias_embedding[idx_i]
            rsd = m_i - m_i_hat

            if FX is not None:
                f_i, _ = get_row(FX, i)
                rsd *= f_i
            if rsd.size > 0:
                 res += rsd.sum()

    out_q.put([process_id, lo, hi, res*mu])

