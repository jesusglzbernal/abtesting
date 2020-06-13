import numpy as np
from scipy import stats

class Test():
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def t_test(self):
        """
        :return:
          t: the t statistic
          2p: the p-value
        """
        N = len(self.a)
        var_a = self.a.var(ddof=1) # ddof to divide var by N-1
        var_b = self.b.var(ddof=1)
        s = np.sqrt( (var_a + var_b) / 2 ) # pulled standard deviation
        t = (self.a.mean() - self.b.mean()) / (s * np.sqrt(2.0/N)) # t statistic
        df = 2 * N - 2 # The degrees of freedom
        p = 1 - stats.t.cdf(t, df=df) # the p-value
        return t, 2*p

    def t_test_builtin(self):
        t, p = stats.ttest_ind(self.a, self.b)
        return t, p

def main():
    N = 10
    a = np.random.randn(N) + 2
    b = np.random.randn(N)

    my_test = Test(a, b)
    t, p = my_test.t_test()
    print("t: ", t, " p: ", p)
    t2, p2 = my_test.t_test_builtin()
    print("t2: ", t2, " p2: ", p2)

if __name__ == '__main__':
    main()