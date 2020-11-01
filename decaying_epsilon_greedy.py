import matplotlib.pyplot as plt
import numpy as np

NUM_TRIALS = 100000
EPS = 0.1
BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]
DEC_RATE = 10000


class Bandit:
    def __init__(self, p):
        # p is the win rate
        self.p = p
        self.p_estimate = 0.0  # The current estimate for the bandit
        self.N = 0  # Number of samples collected so far


    def pull(self):
         # draw a sample with probability p
         return np.random.random() < self.p


    def update(self, x):
         self.N += 1
         self.p_estimate = ((self.N - 1) * self.p_estimate + x) / self.N



def experiment():
    global EPS
    bandits =  [Bandit(p) for p in BANDIT_PROBABILITIES]

    rewards = np.zeros(NUM_TRIALS)
    num_times_exploited = 0
    num_times_explored = 0
    num_optimal = 0
    optimal_j = np.argmax([b.p for b in bandits])
    print("optimal j:", optimal_j)

    for i in range(NUM_TRIALS):
        if np.random.random() < EPS:  #  Use epsilon-greedy to select the next bandit
            num_times_explored += 1
            j = np.random.randint(len(bandits))
        else:
            num_times_exploited += 1
            j = np.argmax([b.p_estimate for b in bandits])

        if j == optimal_j:
            num_optimal += 1

        #  Pull the arm for the bandit with the largest sample
        x = bandits[j].pull()

        #  Update the rewards log
        rewards[i] = x

        #  Update the distribution for the bandit whose arm we just pulled
        bandits[j].update(x)
        if EPS > 0:
            EPS = EPS - 1/DEC_RATE


    #  Print the mean estimates for each bandit
    for b in bandits:
        print("Mean estimate:", b.p_estimate)

    #  Print total reward
    print("Total reward earned:", rewards.sum())
    print("Overall win rate:", rewards.sum() / NUM_TRIALS)
    print("Number of times explored:", num_times_explored)
    print("Number of times exploited:", num_times_exploited)
    print("Number of times selected the optimal bandit:", num_optimal)


    #  Plot the results
    cumulative_rewards = np.cumsum(rewards)
    win_rates = cumulative_rewards / (np.arange(NUM_TRIALS) + 1)
    plt.plot(win_rates)
    plt.plot(np.ones(NUM_TRIALS) * np.max(BANDIT_PROBABILITIES))
    plt.show()


if __name__ == '__main__':
    experiment()