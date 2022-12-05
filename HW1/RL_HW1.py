# import pandas

import random
import matplotlib.pyplot as plt 

class MDP:
    def __init__(self):
        # random.seed(10)
        pass

    def policy(self, state):
        rand = random.uniform(0, 1)
        if (state == "s1" and rand < 0.4) or (state == "s2" and rand < 0.35) or (state == "s3" and rand < 0.9) or (state == "s4" and rand < 0.5) or (state == "s5" and rand < 0.1):
            return "a1"
        else:
            return "a2"
        # policy = {
        #     "s1": "a1" if rand < 0.4 else "a2",
        #     "s2": "a1" if rand < 0.35 else "a2",
        #     "s3": "a1" if rand < 0.9 else "a2",
        #     "s4": "a1" if rand < 0.5 else "a2",
        #     "s5": "a1" if rand < 0.1 else "a2",
        # }
        # return policy

        
    def all_deterministic_policies(self):
        all_policies = []
        for s1 in ["a1", "a2"]:
            for s2 in ["a1", "a2"]:
                for s3 in ["a1", "a2"]:
                    for s4 in ["a1", "a2"]:
                        for s5 in ["a1", "a2"]:
                            all_policies.append({
                                's1': s1,
                                's2': s2,
                                's3': s3,
                                's4': s4,
                                's5': s5,
                            })
        return all_policies
        

    def reward_function(self, state, action):
        if state == "s1":
            return 5 if action == "a1" else 2
        elif state == "s2":
            return -3 if action == "a1" else 7
        elif state == "s3":
            return 3 if action == "a1" else -5
        elif state == "s4":
            return -6 if action == "a1" else 8
        elif state == "s5":
            return 4 if action == "a1" else 10

    def d0(self):
        rand = random.uniform(0, 1)
        if rand < 0.4:
            return "s1"
        else:
            return "s2"

    def transition(self, state, action):
        rand = random.uniform(0, 1)
        if state == "s1":
            if (action == "a1" and rand < 0.1) or (action == "a2" and rand < 0.8):
                return "s3"
            else: 
                return "s4"
        if state == "s2":
            if action == "a1":
                return "s4"
            else:
                return "s5"
        
        
    def runEpisode(self, policy, gamma):
        state_0 = self.d0()
        action_0 = policy(state_0)
        R_0 = self.reward_function(state_0, action_0)
        state_1 = self.transition(state_0, action_0)
        action_1 = policy(state_1)
        R_1 = self.reward_function(state_1, action_1)
        return (R_0 + (gamma*R_1))

    def runEpisode_det(self, policy, gamma):        
        state_0 = self.d0()
        action_0 = policy[state_0]
        R_0 = self.reward_function(state_0, action_0)
        state_1 = self.transition(state_0, action_0)
        action_1 = policy[state_1]
        R_1 = self.reward_function(state_1, action_1)
        return (R_0 + (gamma*R_1))

def main():
    from statistics import mean, variance

    agent = MDP()
    itr = 150000
    
    # 2a
    J = []
    cumulative_G = 0.0
    episodes = []
    for i in range(itr):
        episode = agent.runEpisode(agent.policy, 0.9)
        cumulative_G += episode
        episodes.append(episode)
        J.append(cumulative_G/(i+1))
    plt.plot(list(range(1, itr+1)), J)
    plt.xlabel("Iterations")
    plt.ylabel("\hatJ(pi)")
    plt.show()
    print("J^(pi) = {}".format(round(J[-1], 2)))


    # 2b
    print("Average discounted return Avg(G) = {}".format(round(J[-1], 2)))
    print("Variance of the discounted return Var(G) = {}".format(round(variance(episodes), 4)))
    print("Variance of the average discounted return Var(J(pi)) = {}".format(round(variance(J), 6)))

    # 2c
    for gamma in [0.25, 0.5, 0.75, 0.99]:
        closed_form_J =  3.38 + (gamma*4.5256)
        itr = 150000
        J = []
        cumulative_G = 0.0
        for i in range(itr):
            episode = agent.runEpisode(agent.policy, gamma)
            cumulative_G += episode
            J.append(cumulative_G/(i+1))
        print("Discounted return = {} for gamma = {}".format(round(J[-1], 4), gamma))
        print("Discounted return closed form = {} for gamma = {}".format(round(closed_form_J, 4), gamma))
        assert(abs(round(closed_form_J, 2) - round(J[-1], 2)) < 5e-1)
    # 2d
    policies = agent.all_deterministic_policies()
    best_policy = None
    best_J = -float("inf")
    for policy in policies:
        itr = 350000
        J = []
        cumulative_G = 0.0
        for i in range(itr):
            episode = agent.runEpisode_det(policy, 0.75)
            cumulative_G += episode
            J.append(cumulative_G/(i+1))
        if best_J < J[-1]:
            best_J = J[-1]
            best_policy = policy
    print("best policy = {}, return = {}".format(best_policy, round(best_J, 2)))
if __name__ == '__main__':
    main()
