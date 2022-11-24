from ast import main
import collections
from statistics import mean
import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt

class TD():
    def __init__(self, alpha, gamma, delta):
        self.actions = [(-1,0), (0,1), (1,0), (0,-1)] # up, right, down, left 
        self.arrows = ["↑", "→","↓", "←"]
        self.gamma = gamma
        self.alpha = alpha
        self.delta = delta
        self.v = np.array([[0.0 for j in range(5)] for i in range(5)])
        # self.policy = np.array([["" for j in range(5)] for i in range(5)])
        self.policy = np.array([[(0, 1) for j in range(5)] for i in range(5)])
        # self.states = np.array([[(i, j) for j in range(5)] for i in range(5)])
        self.states = [(i, j) for j in range(5) for i in range(5)]
        self.pi_star = [[(0, 1), (0, 1), (0, 1), (1, 0), (1, 0)],
                   [(0, 1), (0, 1), (0, 1), (1, 0), (1, 0)],
                   [(-1, 0), (-1, 0), (-1, 0), (1, 0), (1, 0)],
                   [(-1, 0), (-1, 0), (-1, 0), (1, 0), (1, 0)],
                   [(-1, 0), (-1, 0), (0, 1), (0, 1), (-1, 0)]]
        self.v_star = [[4.0187,4.5548,5.1576,5.8337,6.4553],
                       [4.3716,5.0324,5.8013,6.6473,7.3907],
                       [3.8672,4.39,  0.0,   7.5769,8.4637],
                       [3.4183,3.8319,0.0,   8.5738,9.6946],
                       [2.9978,2.9309,6.0733,9.6946,0.]]
        # print(self.states)

        self.q = np.array([[[0.0 for a in range(len(self.actions))] for j in range(5)] for i in range(5)])
        self.pi_esoft = collections.defaultdict(list) # key: state, value: list(best action)
        for s in self.states:
            self.pi_esoft[s] = self.actions

        self.test_pol = np.array([[[0.25 for k in range(4)]for j in range(5)] for i in range(5)])
        pass
    
    def transition_function(self):
        counter = 0
        p = defaultdict(list)
        for state in self.states: 
            for next_direction in [(-1,0), (0,1), (1,0), (0,-1), (0,0)]:
                # print("next_direction = {}".format(next_direction[0]))
                # print("state = {}".format(state))
                next_state = (state[0] + next_direction[0], state[1] + next_direction[1]) 
                # print(next_state)
                # print()
                for action in self.actions:
                    
                    prob = 0
                    if ((next_state[0] < 0) or (next_state[1] < 0) or (next_state[0] > 4) or (next_state[1] > 4)):
                        continue
                    if ((state[0] == 2) and (state[1] == 2)) or ((state[0] == 3) and (state[1] == 2)):
                        prob = 0
                        p[state, next_state].append(prob)
                        continue
                    if ((next_state[0] == 2) and (next_state[1] == 2)) or ((next_state[0] == 3) and (next_state[1] == 2)):
                        prob = 0
                        p[state, next_state].append(prob)
                        continue
                    if ((state[0] == 4) and (state[1] == 4)):
                        if ((next_state[0] == 4) and (next_state[1] == 4)):
                            prob = 1
                        else:
                            prob = 0
                        p[state, next_state].append(prob)
                        continue
                    # if ((state[0] == 0) and (state[1] == 2)):
                    #     if ((next_state[0] == 0) and (next_state[1] == 2)):
                    #         prob = 1
                    #     else:
                    #         prob = 0
                    #     p[state, next_state].append(prob)
                    #     continue
                    if action == next_direction:
                        prob = 0.8
                    elif (next_direction == self.actions[(self.actions.index(action) + 1) % 4]) or (next_direction == self.actions[(self.actions.index(action) - 1) % 4]):
                        prob = 0.05
                    elif (next_direction == (0,0)):
                        prob = 0.1
                        if ((state[0] + action[0]) == 2 and (state[1] + action[1]) == 2): # going into the obstacle directly when moving towards the obstacle
                            prob += 0.8
                        if ((state[0] + action[0]) == 3 and (state[1] + action[1]) == 2):# going into the obstacle directly when moving towards the obstacle
                            prob += 0.8
                        if (((state[0] + action[0]) < 0)  or ((state[1] + action[1]) < 0) or ((state[0] + action[0]) > 4)  or ((state[1] + action[1]) > 4)): # going directly into the wall
                            prob += 0.8
                        if (((state[0] + self.actions[(self.actions.index(action) + 1) % 4][0]) == 2) and ((state[1] + self.actions[(self.actions.index(action) + 1) % 4][1]) == 2)): # going into the obstacle mistakenly towrads right
                            prob += 0.05
                        if (((state[0] + self.actions[(self.actions.index(action) + 1) % 4][0]) == 3) and ((state[1] + self.actions[(self.actions.index(action) + 1) % 4][1]) == 2)):# going into the obstacle 
                            prob += 0.05
                        if (((state[0] + self.actions[(self.actions.index(action) - 1) % 4][0]) == 2) and ((state[1] + self.actions[(self.actions.index(action) - 1) % 4][1]) == 2)): # going into the obstacle
                            prob += 0.05
                        if (((state[0] + self.actions[(self.actions.index(action) - 1) % 4][0]) == 3) and ((state[1] + self.actions[(self.actions.index(action) - 1) % 4][1]) == 2)):# going into the obstacle
                            prob += 0.05
                        if ((state[0] + self.actions[(self.actions.index(action) + 1) % 4][0]) < 0): # going into the wall
                            prob += 0.05
                        if ((state[1] + self.actions[(self.actions.index(action) + 1) % 4][1]) < 0): # going into the wall
                            prob += 0.05
                        if ((state[0] + self.actions[(self.actions.index(action) + 1) % 4][0]) > 4): # going into the wall
                            prob += 0.05
                        if ((state[1] + self.actions[(self.actions.index(action) + 1) % 4][1]) > 4): # going into the wall
                            prob += 0.05
                        if ((state[0] + self.actions[(self.actions.index(action) - 1) % 4][0]) < 0): # going into the wall
                            prob += 0.05
                        if ((state[1] + self.actions[(self.actions.index(action) - 1) % 4][1]) < 0): # going into the wall
                            prob += 0.05
                        if ((state[0] + self.actions[(self.actions.index(action) - 1) % 4][0]) > 4): # going into the wall
                            prob += 0.05
                        if ((state[1] + self.actions[(self.actions.index(action) - 1) % 4][1]) > 4): # going into the wall
                            prob += 0.05
                    
                    # print("state = {}, action = {}, next_state = {}, prob = {}".format(state, action, next_state, round(prob, 3)))
                    p[state, next_state].append(round(prob, 3))
        # print(len(p))
        return p
    
    
    def trans_func(self, s, a):
        """
        Args:
            s = tuple(row, col)
            a = action (tuple)
        Returns:
            next state (tuple)
        """
        if s == (4, 4): return s
        rand = random.uniform(0, 1)
        if rand < 0.8:
            s_prime = (s[0] + a[0], s[1] + a[1])
        elif 0.8 < rand <  0.85:
            a = self.actions[(self.actions.index(a) + 1) % 4]
            s_prime = (s[0] + a[0], s[1] + a[1])
        elif  0.85 < rand <  0.9:
            a = self.actions[(self.actions.index(a) - 1) % 4]
            s_prime = (s[0] + a[0], s[1] + a[1])
        else:
            s_prime = s
        if (s_prime == (2,2)) or (s_prime == (3,2)) or (s_prime[0] < 0) or (s_prime[0] > 4) or (s_prime[1] < 0) or (s_prime[1] > 4):
            s_prime = s
        return s_prime





    def reward(self, s, a, s_prime):
        if (s == (4, 4)):
            return 0
        elif s_prime == (4, 4):
            return 10
        elif s_prime == (4, 2):
            return -10
        # elif s_prime == (0, 2):
        #     # return 5
        #     return 4.4844 # found using binary search
        else:
            return 0

    def d0(self):
        states = self.states.copy()
        states.remove((2,2))
        states.remove((3,2))
        random_index = random.randint(0,len(states)-1)
        return states[random_index]




    def run(self, threshold):
        # self.v = np.array([[0 for j in range(5)] for i in range(5)])
        p = self.transition_function()
        count = 0
        while True:
            count += 1
            delta = 0
            v_old = np.copy(self.v)
            # print(np.amax(v_old))
            for s in self.states:
                max_val = -float("inf")
                max_a = None
                for i, a in enumerate(self.actions):
                    val = 0
                    for s_prime in self.states:
                        
                        # print(s, s_prime)
                        try:
                            # print(v_old[s_prime])
                            val += p[s, s_prime][i]*(self.reward(s, a, s_prime) + (self.gamma*v_old[s_prime]))
                        except:
                            continue
                    # print("val = {}".format(val))
                    if max_val < val:
                        max_val = val
                        max_a = i
                # if (s == (1, 1)): print("val = {}".format(val))
                self.v[s] = round(max_val, 4)
                self.policy[s] = self.actions[max_a]
            delta = max(delta, np.amax(abs(self.v - v_old)))
            if delta < threshold:
                break
        return self.v, self.policy, count

    def pi_func(self, pi, s):
        # self.gamma = 0.9
        # v_star, pi_star, iterations = self.run(0.0001)

        return pi[s[0]][s[1]]
    

    def v_star_func(self, s):
        # self.gamma = 0.9
        # v_star, pi_star, iterations = self.run(0.0001)
        return self.v_star[s[0]][s[1]]

    def generateEpisode(self, pi):
        trajectory = []
        s = self.d0()
        while(s != (4, 4)):
            a = self.pi_func(pi, s)
            s_prime = self.trans_func(s, a)
            r = self.reward(s, a, s_prime)
            trajectory.append((s, r))
            s = s_prime
        trajectory.append(((4,4), 0))
        return trajectory

    def pi_esoft_func(self, pi, s, eps):
        """
        Args: 
            pi: dictionary key = states, value = list of actions
            s = tuple(row, col)
            eps = float
        Returns:
            action in state s (tuple)
        """
        # rand = random.uniform(0, 1)
        # A_star = pi[s]
        # A = self.actions
        # A_ = list(set(A) - set(A_star))
        # # print(A_star, A, A_)
        # prob = ((1- eps)/len(A_star)) + (eps/len(A))
        # for i in range(len(A_star)):
        #     if prob*(i) < rand < prob*(i+1):
        #         return A_star[i]
        # for i in range(len(A_)):
        #     if (prob*len(A_star) + (eps/len(A))*(i)) < rand < (prob*len(A_star) + (eps/len(A))*(i+1)):
        #         return A_[i]
        # print(s)
        chosen_action = np.random.choice(4, 1, p=pi[s[0]][s[1]])
        return self.actions[chosen_action[0]]

    def policy_prob(self, s, a, pi, eps):
        A_star = pi[s]
        A = self.actions
        # print(a, A_star)
        if a in A_star:
            # print(((1- eps)/len(A_star)) + (eps/len(A)))
            return ((1- eps)/len(A_star)) + (eps/len(A))
        else:
            # print("else condition")
            # print((eps/len(A)))
            return (eps/len(A))

    def generateEpisode_esoft(self, pi, eps):
        """
        Args:
            pi = dictionary key = states, value = list of actions
            eps
        """
        trajectory = []
        s = self.d0()
        while(s != (4, 4)):
            a = self.pi_esoft_func(pi, s, eps)
            s_prime = self.trans_func(s, a)
            r = self.reward(s, a, s_prime)
            trajectory.append(((s, a), r))
            s = s_prime
        trajectory.append((((4,4), (0, 1)), 0))
        return trajectory


    def td(self,):
        # returns = collections.defaultdict(list)
        max_norm = []
        mse = []
        itr_number = []
        count = 0
        while True:
            count += 1
            episode = self.generateEpisode(self.pi_star)

            for i in range(0, len(episode) - 1):
                # a = self.pi_func(self.pi_star, episode[i][0])
                self.v[episode[i][0][0]][episode[i][0][1]] = \
                self.v[episode[i][0][0]][episode[i][0][1]] + \
                self.alpha * (episode[i][1] + self.gamma * self.v[episode[i+1][0][0]][episode[i+1][0][1]] - self.v[episode[i][0][0]][episode[i][0][1]])
            max_norm.append(np.amax(abs(self.v - self.v_star)))
            if count % 250 == 0:
                mse.append(self.mse(self.v, self.v_star))
                itr_number.append(count)

            if np.amax(abs(self.v - self.v_star)) < self.delta:
                break
        # print("max norm = {}".format(max_norm[-1]))
        # print("Iterations to converge = {}".format(count))
        # # plt.plot(max_norm)
        # # plt.title("Max norm")
        # # plt.xlabel("Iterations")
        # # plt.ylabel("Max norm")
        # print("MSE = {}".format(mse[-1]))
        # # plt.plot(max_norm)
        # # plt.title("Plotting Max norm")

        # plt.plot(itr_number, mse)
        # # plt.title("Mean squared Error for eps = {}".format(eps))
        # plt.title("Mean squared Error for alpha = {}".format(self.alpha))
        # plt.xlabel("Iterations")
        # plt.ylabel("MSE")
        # plt.show()
        return count
    


    def every_visit(self, ):
        returns = collections.defaultdict(list)
        max_norm = []
        count = 0
        while True:
            count += 1
            episode = self.generateEpisode(self.pi_star)
            # print("episode = {}".format(episode))
            states_present = []
            rewards = []
            for s, r in episode:
                states_present.append(s)
                rewards.append(r)
            for i, s in enumerate(states_present):
                G = 0
                temp_rewards = rewards[i:]
                # print("temp_rewards = {}".format(temp_rewards))
                for pow in range(len(temp_rewards)):
                    G += (self.gamma**pow) * temp_rewards[pow]
                # print(G)
                returns[s].append(G)
                # print(returns)
                self.v[s[0]][s[1]] = mean(returns[s])
            max_norm.append(np.amax(abs(self.v - self.v_star)))
            if np.amax(abs(self.v - self.v_star)) < 0.1 or count > 10000:
                break
        print("max norm = {}".format(max_norm[-1]))
        print("Iterations to converge = {}".format(count))
        plt.plot(max_norm)
        plt.title("Max norm")
        plt.xlabel("Iterations")
        plt.ylabel("Max norm")
        plt.show()
        return count
 
    def e_soft(self, eps, decay=False):
        returns = collections.defaultdict(list)
        max_norm = []
        count = 0
        mse = []
        itr_number = []
        while True and eps > 0:
            count += 1
            if decay and count % 500 == 0:
                eps -= 0.05
            # episode = self.generateEpisode_esoft(self.pi_esoft, eps)
            episode = self.generateEpisode_esoft(self.test_pol, eps)
            # print("episode = {}".format(episode))
            states_action_present = []
            rewards = []
            for s_a, r in episode:
                states_action_present.append(s_a)
                rewards.append(r)
            for s_a in set(states_action_present):
                first_index = states_action_present.index(s_a)
                G = 0
                temp_rewards = rewards[first_index:]
                for pow in range(len(temp_rewards)):
                    G += (self.gamma**pow) * temp_rewards[pow]
                # print("state= {}, action = {}".format(s_a[0], s_a[1]))
                returns[s_a].append(G)
                index_a = self.actions.index(s_a[1])
                row = s_a[0][0]
                col = s_a[0][1]
                # print("q update value = {}".format(mean(returns[s_a])))
                self.q[row][col][index_a] = mean(returns[s_a]) # Update q
                # print("udated q value = {}".format(self.q[row][col][index_a]))
                best_a_list = []
                # print("list of best actions b4 = {}".format(best_a_list))
                best_qsa = -float("inf")
                # for i, expl_a in enumerate(self.actions):
                #     if best_qsa < self.q[row][col][i]:
                #         best_qsa = self.q[row][col][i]
                #         best_a_list = [expl_a]
                #     elif best_qsa == self.q[row][col][i]:
                #         best_a_list.append(expl_a)
                # self.pi_esoft[s_a[0]] = best_a_list
                
                for i, expl_a in enumerate(self.actions):
                    if best_qsa < self.q[row][col][i]:
                        best_qsa = self.q[row][col][i]
                        best_a_list = [i]
                    elif best_qsa == self.q[row][col][i]:
                        best_a_list.append(i)
                # print("list of best actions after = {}".format(best_a_list))
                not_best_list = list(set(range(4)) - set(best_a_list))
                new_prob = max(0, ((1- eps)/len(best_a_list)) + (eps/len(self.actions)))
                remaining_prob = (eps/len(self.actions))
                np.put(self.test_pol[row][col], best_a_list, [new_prob]*len(best_a_list))
                np.put(self.test_pol[row][col], not_best_list, [remaining_prob]*len(not_best_list))
            for s in self.states:
                # self.v[s[0]][s[1]] = sum([self.policy_prob(s, a, self.pi_esoft, eps)*self.q[s[0]][s[1]][a_index] for a_index, a in enumerate(self.actions)])
                self.v[s[0]][s[1]] = sum([self.test_pol[s[0]][s[1]][a_index]*self.q[s[0]][s[1]][a_index] for a_index, a in enumerate(self.actions)])
            max_norm.append(np.amax(abs(self.v - self.v_star)))
            # if np.amax(abs(self.v - self.v_star)) < 0.1:
            if count % 250 == 0:
                mse.append(self.mse(self.v, self.v_star))
                itr_number.append(count)

            if count > 10000:
                break
        print("max norm = {}".format(np.amax(abs(self.v - self.v_star))))
        print("MSE = {}".format(mse[-1]))
        # plt.plot(max_norm)
        # plt.title("Plotting Max norm")

        plt.plot(itr_number, mse)
        # plt.title("Mean squared Error for eps = {}".format(eps))
        plt.title("Mean squared Error for eps = {}".format("decaying"))
        plt.xlabel("Iterations")
        plt.ylabel("MSE")

        plt.show()
        # print(returns)


        pass

    def mse(self, m1, m2):
        return np.square(np.subtract(m1, m2)).mean() 

def main():
    def replace(inp, positions, char):
        for pos in positions:
            inp[pos] = char
    obstacles = [(2,2), (3,2)]
    goal = [(0,2), (4,4)]

    gamma = 0.9
    alpha = 0.2
    delta = 0.1

    print("running Temporal Difference Learning")
    iterations_count = []
    alpha_val = []
    count = 1
    while alpha > 0.01:
        alpha -= 0.01

        alpha = round(alpha, 4)
        td = TD(alpha=alpha, gamma=gamma, delta=delta)
        iterations_count.append(td.td())
        alpha_val.append(alpha)
        print(td.v)
        count += 1
    plt.plot(alpha_val, iterations_count)

if __name__ == '__main__':
    main()