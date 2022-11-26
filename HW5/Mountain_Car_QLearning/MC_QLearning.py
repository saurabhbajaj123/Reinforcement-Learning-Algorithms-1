from ast import main
import collections
from statistics import mean
import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import matplotlib.lines as mlines
from math import cos, pi, sin, sqrt
import statistics

class QLearning:
    def __init__(self, alpha, gamma, delta, num_bins):
        self.gamma = gamma
        self.alpha = alpha
        self.delta = delta

        self.bins_x = np.linspace(start=-1.2, stop=0.5, num=num_bins)
        self.bins_v = np.linspace(start=-0.07, stop=0.07, num=num_bins)

        self.states = [(i, j) for j in range(num_bins + 1) for i in range(num_bins + 1)]
        self.actions = [-1, 1]
        self.v = np.array([[0.0 for j in range(num_bins + 1)] for i in range(num_bins + 1)])
        self.q = np.array([[[0.0 for a in range(len(self.actions))] for j in range(num_bins + 1)] for i in range(num_bins + 1)])
        self.test_pol = np.array([[[float(1/len(self.actions)) for k in range(len(self.actions))]for j in range(num_bins + 1)] for i in range(num_bins + 1)])


    def trans_func(self, s, a):
        x_t, v_t = s
        v_t_prime = v_t + 0.001*a - 0.0025*cos(3*x_t)
        x_t_prime = x_t + v_t_prime

        if x_t_prime < -1.2:
            x_t_prime = -1.2
        elif x_t_prime > 0.5:
            x_t_prime = 0.5
        
        if v_t_prime < -0.07:
            v_t_prime = -0.07
        elif v_t_prime > 0.07:
            v_t_prime = 0.07
        
        if x_t_prime == -1.2 or x_t_prime == 0.5:
            v_t_prime = 0

        # x_t_prime = np.digitize([x_t_prime], self.bins)[0]
        # v_t_prime = np.digitize([v_t_prime], self.bins)[0]
        return (x_t_prime, v_t_prime)
    
    def pi_func(self, pi, s):
        row = np.digitize([s[0]], self.bins_x)[0]
        col = np.digitize([s[1]], self.bins_v)[0]
        chosen_action = np.random.choice(len(self.actions), 1, p=pi[row][col])
        return self.actions[chosen_action[0]]

    def d0(self):
        # s_0 = random.uniform(-0.6, -0.4)
        s_0 = np.random.normal(loc=-0.3, scale=0.1)
        return (s_0, 0)

    def reward_function(self, state):
        if state[0] == 0.5:
            return 0
        else:
            return -1
    
    def e_soft_policy_update(self, s, eps):
        row = np.digitize([s[0]], self.bins_x)[0]
        col = np.digitize([s[1]], self.bins_v)[0]

        best_a_list = []
        best_qsa = -float("inf")
        
        for i, expl_a in enumerate(self.actions):
            if best_qsa < self.q[row][col][i]:
                best_qsa = self.q[row][col][i]
                best_a_list = [i]
            elif best_qsa == self.q[row][col][i]:
                best_a_list.append(i)

        not_best_list = list(set(range(len(self.actions))) - set(best_a_list))
        new_prob = max(0, ((1- eps)/len(best_a_list)) + (eps/len(self.actions)))
        remaining_prob = (eps/len(self.actions))
        np.put(self.test_pol[row][col], best_a_list, [new_prob]*len(best_a_list))
        np.put(self.test_pol[row][col], not_best_list, [remaining_prob]*len(not_best_list))

    def softmax_policy_update(self, s, sigma):
        row = np.digitize([s[0]], self.bins_x)[0]
        col = np.digitize([s[1]], self.bins_v)[0]
        p = sigma*self.q[row][col]
        self.test_pol[row][col] = np.exp(p - max(p))/sum(np.exp(p - max(p)))


    def update_q(self, s, s_prime, a, r):
        # x_t_prime = np.digitize([x_t_prime], self.bins_x)[0]
        # v_t_prime =

        row = np.digitize([s[0]], self.bins_x)[0]
        col = np.digitize([s[1]], self.bins_v)[0]
        index_a = self.actions.index(a)
        # next_a = self.pi_func(pi=self.test_pol, s=s_prime)
        next_row = np.digitize([s_prime[0]], self.bins_x)[0]  
        next_col = np.digitize([s_prime[1]], self.bins_v)[0]
        # index_next_a = self.actions.index(next_a)
        self.q[row][col][index_a] = self.q[row][col][index_a] \
            + self.alpha * (r + (self.gamma * np.amax(self.q[next_row][next_col])) - self.q[row][col][index_a]) 


    def qlearning(self, eps):
        count = 0
        num_actions = 0
        num_episodes_list = []
        num_actions_list = []
        num_steps_list = []
        while True:
            count += 1
            s = self.d0()
            num_steps = 0
            while (s[0] != 0.5):
                num_steps += 1
                num_actions += 1
                a = self.pi_func(self.test_pol, s)
                s_prime = self.trans_func(s, a)
                r = self.reward_function(s)

                self.update_q(s, s_prime, a, r)

                self.e_soft_policy_update(s=s, eps=min(eps, 1/count))
                # self.softmax_policy_update(s=s, sigma=count)

                s = s_prime
            # print()
            # print("episode = {}, num actions till now = {}, num steps for this episode = {}".format(count, num_actions, num_steps))
            num_episodes_list.append(count)
            num_actions_list.append(num_actions)
            num_steps_list.append(num_steps)
            
            self.v = np.max(self.q, axis=2)

            if count > 50:
                break
                
        return count, num_episodes_list, num_actions_list, num_steps_list


def main():
    gamma = 0.9
    alpha = 0.01
    delta = 0.1
    sigma = 1
    num_bins = 5
    eps = 0.01

    num_acts_mean_list = []
    num_steps_mean = []
    for _ in tqdm(range(10)):
        ql = QLearning(alpha=alpha, gamma=gamma, delta=delta, num_bins=num_bins)
        count, num_episodes_list, num_actions_list, num_steps_list = ql.qlearning(eps=eps)

        plt.figure(0)
        plt.plot(num_actions_list, num_episodes_list, 'c')
        plt.title("Learning curve")
        plt.xlabel("Time Steps")
        plt.ylabel("Episodes")
        num_acts_mean_list.append(num_actions_list)

        # plt.figure(1)
        # plt.plot(num_episodes_list, num_steps_list, 'c')
        # plt.title("Learning curve")
        # plt.xlabel("Time Steps")
        # plt.ylabel("Episodes")
        num_steps_mean.append(num_steps_list)

    column_average = [sum(sub_list) / len(sub_list) for sub_list in zip(*num_acts_mean_list)]

    plt.figure(0)
    plt.plot(column_average, num_episodes_list, 'k')
    plt.title("Learning curve")
    plt.xlabel("Time Steps")
    plt.ylabel("Episodes")
    # eight = mlines.Line2D([], [], color='c', marker='s', ls='', label='')
    nine = mlines.Line2D([], [], color='k', marker='s', ls='', label='mean')
    plt.legend(handles=[nine])

    column_average_steps = [sum(sub_list) / len(sub_list) for sub_list in zip(*num_steps_mean)]
    column_variance_steps = [statistics.stdev(i) for i in zip(*num_steps_mean)]

    plt.figure(1)
    plt.plot(num_episodes_list, column_average_steps, 'k')
    # plt.fill_between(column_average_steps, list(map(float.__add__, column_average_steps, column_variance_steps)), list(map(float.__sub__, column_average_steps, column_variance_steps)), facecolor='blue', alpha=0.5)
    plt.errorbar(num_episodes_list, column_average_steps, yerr=column_variance_steps, elinewidth=0.75, linestyle='None', marker='.')
    plt.title("Learning curve")
    plt.ylabel("Number of Steps")
    plt.xlabel("Episodes")
    # eight = mlines.Line2D([], [], color='c', marker='s', ls='', label='')
    nine = mlines.Line2D([], [], color='k', marker='s', ls='', label='mean')
    plt.legend(handles=[nine])

    plt.show()

if __name__ == '__main__':
    main()