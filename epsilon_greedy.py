"""
Author     : Jesus Gonzalez, using code from the Lazy Programmer with some modifications
Date       : September 11, 2020
Description: This program shows how the multi-armed bandit works with a binary problem
             We can have M Bandits and the outcome in each sample is 0 or 1
             The experiment runs for NUM_TRIALS trials (iterations)
             The probability for exploitation/exploration is given by EPS (epsilon)
             The actual win probabilities of the Bandits are stored in BANDIT_PROBABILITIES
"""

import matplotlib.pyplot as plt
import numpy as np

NUM_TRIALS = 100000
EPS = 0.1
BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]


# The Bandit class
class Bandit:
    def __init__(self, p):
        """
        Function   : Constructor
        Description: Initialize the Bandit
        Inputs     :
            p: The actual win rate for the Bandit
        Attributes :
            p_estimate: The estimate of the winning rate of the Bandit
            N: The number of times that this Bandit has been used (number of samples collected)
        """
        self.p = p
        self.p_estimate = 0.0  # The current estimate for the bandit
        self.N = 0  # Number of samples collected so far


    def pull(self):
        """
        Function   : pull
        Description: Take a sample from this bandit using its actual win rate
        Inputs     : None
        Output     : The sample, either 0 or 1 (False or True)
        """
        # draw a sample with probability p
        return np.random.random() < self.p


    def update(self, x):
        """
        Function   : update
        Description: Updates the probability distribution of the estimation of the win rate for the Bandit, p_estimate
                     and the number of samples collected for this Bandit
        Inputs     : x, the new sample used to update the probability distribution of the bandit
        Output     : None
        Note       : The update function computes the mean of the estimate in an optimized way
        """
        self.N += 1
        self.p_estimate = ((self.N - 1) * self.p_estimate + x) / self.N



def experiment():
    bandits = [Bandit(p) for p in BANDIT_PROBABILITIES]

    rewards = np.zeros(NUM_TRIALS)  # Initialize the rewards vector
    num_times_exploited = 0  # Number of times we chose to exploit
    num_times_explored = 0  # Number of times we chose to explore
    num_optimal = 0  # Number of times the optimal Bandit was chosen
    optimal_j = np.argmax([b.p for b in bandits])  # The optimal Bandit
    print("optimal j:", optimal_j)

    for i in range(NUM_TRIALS):  # Loop to perform the trials
        if np.random.random() < EPS:  # Use epsilon-greedy to select the next bandit
            num_times_explored += 1  # We chose to explore
            j = np.random.randint(len(bandits))  # Randomly selected a Bandit
        else:
            num_times_exploited += 1  # We chose to exploit
            j = np.argmax([b.p_estimate for b in bandits])  # We choose the best "so far" Bandit

        if j == optimal_j:
            num_optimal += 1  # Keep track of the number of times we choose the optimal Bandit

        #  Pull the arm for the bandit with the largest sample
        x = bandits[j].pull()

        #  Update the rewards log
        rewards[i] = x

        #  Update the distribution for the bandit whose arm we just pulled
        bandits[j].update(x)

    # Print the mean estimates for each bandit
    for b in bandits:
        print("Mean estimate:", b.p_estimate)

    #  Print total reward
    print("Total reward earned:", rewards.sum())
    print("Overall win rate:", rewards.sum() / NUM_TRIALS)
    print("Number of times explored:", num_times_explored)
    print("Number of times exploited:", num_times_exploited)
    print("Number of times selected the optimal bandit:", num_optimal)

    # Plot the results
    cumulative_rewards = np.cumsum(rewards)
    win_rates = cumulative_rewards / (np.arange(NUM_TRIALS) + 1)
    plt.plot(win_rates)
    plt.plot(np.ones(NUM_TRIALS) * np.max(BANDIT_PROBABILITIES))
    plt.show()


if __name__ == '__main__':
    experiment()
