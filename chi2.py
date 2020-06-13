import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2
import matplotlib.pyplot as plt

class DataGenerator:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

    def next(self):
        click1 = 1 if (np.random.random() <= self.p1) else 0
        click2 = 1 if (np.random.random() <= self.p2) else 0
        return click1, click2

class Chi_2:
    def __init__(self, T):
        self.T = T

    def updateT(self, T):
        self.T = T

    def chi_2(self):
        det = self.T[0, 0] * self.T[1, 1] - self.T[0, 1] * self.T[1, 0]
        c2 = float(det) / self.T[0].sum() * det / self.T[1].sum() * self.T.sum() / self.T[:, 0].sum() / self.T[:,1].sum()
        p = 1 - chi2.cdf(x=c2, df=1)
        return p

def chi2_experiment(p1, p2, N):
    data = DataGenerator(p1, p2)

    p_values = np.empty(N)
    T = np.zeros((2, 2)).astype(np.float32)
    ch2 = Chi_2(T)
    for i in range(N):
        c1, c2 = data.next()
        T[0, c1] += 1
        T[1, c2] += 1
        ch2.updateT(T)
        if i < 10:
            p_values[i] = None
        else:
            p_values[i] = ch2.chi_2()
    plt.plot(p_values)
    plt.plot(np.ones(N)*0.05)
    plt.show()

    
chi2_experiment(0.1, 0.11, 20000)