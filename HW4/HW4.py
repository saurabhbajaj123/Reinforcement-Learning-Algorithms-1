from ast import main
from email.policy import default
import numpy as np
import math
import random
from collections import defaultdict

class ValueIteration():
    def __init__(self, gamma):
        self.actions = [(-1,0), (0,1), (1,0), (0,-1)] # up, right, down, left 
        self.arrows = ["↑", "→","↓", "←"]
        self.gamma = gamma
        self.v = np.array([[0.0 for j in range(5)] for i in range(5)])
        self.policy = np.array([["" for j in range(5)] for i in range(5)])
        # self.states = np.array([[(i, j) for j in range(5)] for i in range(5)])
        self.states = [(i, j) for j in range(5) for i in range(5)]
        # print(self.states)
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
                    if ((state[0] == 0) and (state[1] == 2)):
                        if ((next_state[0] == 0) and (next_state[1] == 2)):
                            prob = 1
                        else:
                            prob = 0
                        p[state, next_state].append(prob)
                        continue
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
    
    def reward(self, s, a, s_prime):
        if (s == (4, 4)) or (s == (0 , 2)):
            return 0
        elif s_prime == (4, 4):
            return 10
        elif s_prime == (4, 2):
            return -10
        elif s_prime == (0, 2):
            # return 5
            return 4.4844 # found using binary search
        else:
            return 0

    def d0(self):
        return (0,0)

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
                self.policy[s] = self.arrows[max_a]
            delta = max(delta, np.amax(abs(self.v - v_old)))
            if delta < threshold:
                break
        return self.v, self.policy, count

def main():
    def replace(inp, positions, char):
        for pos in positions:
            inp[pos] = char
    obstacles = [(2,2), (3,2)]
    goal = [(0,2), (4,4)]

    # gamma = 0.9133
    gamma = 0.9
    val_iter = ValueIteration(gamma=gamma)
    # print(val_iter.transition_function())
    ans = val_iter.run(0.0001)
    value = ans[0]
    policy = ans[1]
    iterations = ans[2]
    replace(policy, obstacles, "")
    replace(policy, goal, "G")
    print("value")
    # print(value)
    print(np.array_str(value, precision = 4, suppress_small=True))
    print("policy")
    print(policy)
    print("stopped in {} iterations".format(iterations))


    # val_iter = ValueIteration(gamma=0.25)
    # # print(val_iter.transition_function())
    # ans = val_iter.run(0.0001)
    # value = ans[0]
    # policy = ans[1]
    # iterations = ans[2]
    # replace(policy, obstacles, "")
    # replace(policy, goal, "G")
    # print("value")
    # # print(value)
    # print(np.array_str(value, precision = 4, suppress_small=True))

    # print("policy")
    # print(policy)
    # print("stopped in {} iterations".format(iterations))

if __name__ == '__main__':
    main()
    