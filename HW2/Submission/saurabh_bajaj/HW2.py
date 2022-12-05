
import random
from math import cos, pi, sin, sqrt
from statistics import mean, stdev
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

class MountainCar:
    def __init__(self, M=2):
        self.count = 0
        self.gamma = 1
        self.M = M
        pass
    
    def policy(self, state, theta):
        M = self.M
        state = ((state[0] - (-1.2))/(0.5 - (-1.2)), (state[1] - (-0.07))/(0.07 - (-0.07)))
        phi = np.array([1] + [cos(i*pi*state[0]) for i in range(1, M+1)] + [cos(i*pi*state[1]) for i in range(1, M+1)]).reshape(2*M+1, 1)
        # state = (2*(state[0] - (-1.2))/(0.5 - (-1.2)) - 1, 2*(state[1] - (-0.07))/(0.07 - (-0.07)) - 1)
        # phi = np.array([1] + [sin(i*pi*state[0]) for i in range(1, M+1)] + [sin(i*pi*state[1]) for i in range(1, M+1)]).reshape(2*M+1, 1)
        if np.dot(phi.T, theta)[0] <= 0:
            return -1
        else:
            return 1

    def transition(self, state, action):
        x_t, v_t = state
        v_t_prime = v_t + 0.001*action - 0.0025*cos(3*x_t)
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

        return (x_t_prime, v_t_prime)

    def d0(self):
        s_0 = random.uniform(-0.6, -0.4)
        # s_0 = np.random.normal(loc=-0.3, scale=0.1)
        return (s_0, 0)
    
    def reward_function(self, state):
        if state[0] == 0.5:
            return 0
        else:
            return -1

    
    def estimate_J(self, theta_k, N):
        G = 0
        for i in range(N):
            G += self.runEpisode(theta_k)
        return G/N
    
    def runEpisode(self, theta_k):
        reward = 0
        state = self.d0()
        for t in range(1000):
            action = self.policy(state, theta_k)
            R = self.reward_function(state)
            next_state = self.transition(state, action)
            reward += (self.gamma**t) * R
            if next_state[0] == 0.5:
                break
            state = next_state
        return reward


    def BBO_CEM(self, theta, sigma, K, Ke, N, eps):
        warm_up = 20
        cov_mat = sigma*np.identity(2*self.M + 1)

        mean_J_list = []
        # stddev_J_list = []
        
        difference_list = []
        for iterations in tqdm(range(20)):
            h = []
            for k in range(K):
                theta_k = np.random.multivariate_normal(theta, cov_mat)
                J_hat = self.estimate_J(theta_k, N)
                h.append((J_hat, theta_k))
            temp_theta = 0
            # temp_J = 0
            theta_k_list = []

            for y in sorted(h, key=lambda x: x[0], reverse=True)[:Ke]:
                temp_theta += y[1]
                theta_k_list.append(y[1] - theta)
                # temp_J += y[0]

            theta_new = temp_theta/(eps + Ke)
            cov_mat = (eps*np.identity(2*self.M + 1) + sum([np.dot(theta_k_list[i].reshape(2*self.M + 1, 1), theta_k_list[i].reshape(1, 2*self.M + 1)) for i in range(len(theta_k_list))]))/(eps+Ke)
            difference = np.linalg.norm((theta_new - theta).reshape(2*self.M + 1, 1), ord='fro')
            difference_list.append(difference)

            J_mean = self.estimate_J(theta_new, N)
            mean_J_list.append(J_mean)

            if iterations > warm_up:
                if difference < 1e-4:
                    print("breaking because of theta")
                    break
                if len(mean_J_list) >1 and abs(mean_J_list[-1] - mean_J_list[-2]) < 1e-4:
                    print("breaking because of J")
                    break
            theta = theta_new

        print("last J = {}".format(mean_J_list[-1]))
        # plt.figure()
        # plt.ylim(-1050, 0)
        # plt.plot(mean_J_list)
        # plt.savefig("sigma = {}, K = {}, Ke = {}, M = {}, N = {}".format(int(sigma*100), K, Ke, self.M, N))

        return mean_J_list
def main():
    
    # agent.d0()
    # print(np.dot(np.arange(3*4*5*6).reshape((3,4,5,6)),np.arange(3*4*5*6)[::-1].reshape((5,4,6,3))).shape)
    highest_J = -float("inf")
    best_hyperparameters = None
    # sigma_range = list(random.sample(range(1, 10), 3))
    # sigma_range = list(np.random.random(2))
    # sigma_range = [0.1, 0.6, 2, 10, 50, 200]
    # sigma_range = [10, 15]
    sigma_range = [15]
    # # K_range = list(random.sample(range(250, 251), 1))
    # # K_range = [20, 50, 90, 150, 230, 330]
    # # K_range = [120, 150, 170]
    # # K_range = [150, 170, 190]
    K_range = [170]
    # # M_range = list(random.sample(range(4, 5), 1))
    # # M_range = [2, 4, 6, 8]
    # # M_range = [4, 6, 8, 10, 12, 14]
    # # M_range = [4, 6, 8]
    M_range = [6]
    # Ke_range = random.sample(range(5, int(sqrt(K))), 3)
    # Ke_range = [5, 10, 50]
    Ke_range = [50]
    
    for sigma in sigma_range:
        for K in K_range:
            for Ke in Ke_range:
                for M in M_range:
                    for N in range(10, 11, 1):
                        start = time.time()
                        # sigma = 400
                        # K = 250
                        # Ke = 5
                        # N = 10
                        eps = 0.000001
                        # M = 4
                        print("sigma = {}, K = {}, Ke = {}, M = {}, N = {}".format(round(sigma, 2), K, Ke, M, N))
                        agent = MountainCar(M)
                        # theta = np.random.normal(loc=0.0, scale=1.0, size=2*M+1)
                        theta = np.zeros(2*M+1)
                        J_new = agent.BBO_CEM(theta, round(sigma, 2), K, Ke, N, eps)
                        if highest_J < J_new[-1]:
                            highest_J = J_new[-1]
                            best_hyperparameters = (round(sigma, 2), K, Ke, M, N)
                        print("Time taken = {}".format(time.time() - start))


    print("highest J = {}".format(highest_J))
    print("Best parameterts = {}".format(best_hyperparameters))

    # Plotting the mean and variance for 5 runs of CE
    # hyper_para_range = [
    #     [10, 170, 50, 6, 10],
    #     [15, 170, 50, 6, 10],
    #     [15, 170, 50, 4, 10],
    #     [15, 170, 5, 6, 10],
    # ]
    
    # for i, hyp in enumerate(hyper_para_range):
    #     CE_trials = []
    #     for trials in range(20):
    #         # sigma = 10
    #         # K = 50
    #         # Ke = 5
    #         # M = 4
    #         # N = 10
    #         eps = 0.000001
    #         sigma, K, Ke, M, N = hyp
    #         print("sigma = {}, K = {}, Ke = {}, M = {}, N = {}".format(sigma, K, Ke, M, N))
    #         agent = MountainCar(M)
    #         # theta = np.random.normal(loc=0.0, scale=1.0, size=2*M+1)
    #         theta = np.zeros(2*M+1)
    #         J_new = agent.BBO_CEM(theta, round(sigma, 2), K, Ke, N, eps)
    #         CE_trials.append(J_new)
    #     CE_trials_arr = np.array(CE_trials)
    #     mean_ = np.mean(CE_trials_arr, axis = 0)
    #     std_ = np.std(CE_trials_arr, axis = 0, ddof=1)

    #     plt.figure()
    #     plt.errorbar(x = list(range(len(mean_))), y=mean_, yerr=std_)
    #     plt.xlabel('Iterations')
    #     plt.ylabel('Return')
    #     plt.title("sigma = {}, K = {}, Ke = {}, M = {}, N = {}".format(sigma, K, Ke, M, N))
    #     plt.savefig("Setting {}".format(i))

    #     print("average converged J = {}".format(mean_[-1]))
    #     print("converged variance = {}".format(std_[-1]))

if __name__ == '__main__':
    main()