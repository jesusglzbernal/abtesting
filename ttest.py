import numpy as np
import pandas as pd
from scipy import stats

class Test():
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def t_test(self, equal_var=True):
        """
        :return:
          t: the t statistic
          2p: the p-value
        """
        N = len(self.a)
        var_a = self.a.var(ddof=1) # ddof to divide var by N-1
        var_b = self.b.var(ddof=1)
        if equal_var: # Paired Student's t-test
            s = np.sqrt((var_a + var_b) / 2)  # pulled standard deviation
            t = (self.a.mean() - self.b.mean()) / (s * np.sqrt(2.0/N))  # t statistic
            df = 2 * N - 2  # The degrees of freedom
            p = 1 - stats.t.cdf(t, df=df)  # the p-value
        else:  # Unpaired or Welch's t-test: Unequal sample size or unequal variance
            N1 = len(self.a)
            N2 = len(self.b)
            s1_sq = self.a.var()
            s2_sq = self.b.var()
            t = (self.a.mean() - self.b.mean()) / np.sqrt(s1_sq / N1 + s2_sq / N2)
            nu1 = N1 - 1
            nu2 = N2 - 1
            df = (s1_sq / N1 + s2_sq / N2) ** 2 / ((s1_sq * s1_sq) / (N1 * N1 * nu1) + (s2_sq * s2_sq) / (N2 * N2 * nu2))
            p = (1 - stats.t.cdf(np.abs(t), df=df))
        return t, 2*p

    def t_test_builtin(self, equal_var=True):
        t, p = stats.ttest_ind(self.a, self.b, equal_var=equal_var)
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


def ttest_exercise():
    df = pd.read_csv("~/python/machine_learning_examples/ab_testing/advertisement_clicks.csv")
    df_a = df[df['advertisement_id'] == 'A']
    df_b = df[df['advertisement_id'] == 'B']
    a = df_a['action']
    b = df_b['action']
    my_test = Test(a, b)
    t, p = my_test.t_test(equal_var=False)
    print("t:\t", t, "p:\t ", p)
    t2, p2 = my_test.t_test_builtin(equal_var=False)
    print("t2:\t", t2, "p2\t: ", p2)
    t3, p3 = stats.ttest_ind(a, b, equal_var=False)
    print("t3:\t", t3, " p3:\t", p3)





ttest_exercise()

'''
if __name__ == '__main__':
    main()
'''
