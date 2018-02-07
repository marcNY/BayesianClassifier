import numpy as np
import math
import collections
from collections import Counter
from functools import partial


class BayesClassifier():
    def __init__(self, X_train, Y_train):
        self.X = X_train
        self.y = Y_train
        self.occ = Counter(self.y)
        self.prior = self._compute_prior()
        self.mu = self._compute_mu()
        self.sig, self.siginv = self._compute_sig()

    @property
    def classes(self):
        return list(self.occ.keys())

    @property
    def n_train(self):
        return self.y.shape[0]

    ## Training Algorithms

    def _compute_prior(self):
        A = {k: float(v) / float(self.n_train) for k, v in self.occ.items()}
        assert(sum(A.values())==1)
        return A

    def _compute_mu(self):
        f = partial(BayesClassifier._compute_mui, self.X, self.y)
        return {k: f(k) for k in self.classes}

    def _compute_sig(self):
        f = partial(BayesClassifier._compute_sigi, self.X, self.y)
        A = {k: f(k) for k in self.classes}
        AINV = {k: np.linalg.inv(v) for k, v in A.items()}
        return A, AINV

    @staticmethod
    def _compute_mui(X, y, i):
        M = X[y == i]
        return np.sum(M, axis=0) / float((y == i).sum())

    @staticmethod
    def _compute_sigi(X, y, i):
        identity = y == i
        M = X - BayesClassifier._compute_mui(X, y, i)
        M = M[identity]
        Ny = float((y == i).sum())
        SIG = np.dot(np.transpose(M), M) / Ny
        return SIG

    ## Prediction Algorithm

    def _compute_probai(self, X0, c):
        mu = self.mu[c]
        sig = self.sig[c]
        siginv = self.siginv[c]
        pi = self.prior[c]
        A = pi / math.sqrt(np.linalg.det(sig))
        B = np.dot(X0 - mu, siginv)
        C = np.dot(B, np.transpose(X0 - mu))
        P = math.exp(-1 / 2 * C) * A
        return P

    def _compute_p(self, X0):
        P= {k: self._compute_probai(X0, k) for k in self.classes}
        return P

    def _compute_prob(self, Xr):
        d = self._compute_p(Xr)
        od = collections.OrderedDict(sorted(d.items()))
        l = list(od.values())
        return np.asarray(l)

    def classify(self, Xtest):
        nclass = len(self.classes)
        npred = Xtest.shape[0]
        P = np.zeros([npred, nclass])
        for i in range(0, npred - 1):
            p=self._compute_prob(Xtest[i])
            P[i] = p/np.sum(p)
        print(P.size)
        return P

    def __str__(self):
        s = "Bayes Classifier: \n the classes are %s\n they appear %s\n the means of the classifier are: %s"
        return s % (self.classes, self.occ, self.mu)
