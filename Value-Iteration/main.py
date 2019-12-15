import numpy as np 
import sys
sys.path.append("../")
from frozen_lake_env import *

class ValueIterationLearning:
    def __init__(self, number_of_states, gamma):
        self.env = FrozenLakeEnv()
        self.reset_env()
        # learning parameters
        self.actions = {'LEFT':0, 'RIGHT':2, 'UP':3, 'DOWN':1}
        self.number_of_states = number_of_states
        self.values = np.zeros(number_of_states)
        self.gamma = gamma
    
    def reset_env(self):
        self.env.reset()
        
    def calculate_new_state_value(self, state):
        max_value, max_value_action = None, None
        
        for a in self.actions.keys():
            new_states_and_probs = self.env.P[state][self.actions[a]]
            
            sigma = 0
            # (p, s prim, reward, is goal state)
            for nsp in new_states_and_probs:
                new_state = nsp[1]
                probability_of_transition = nsp[0]
                reward_of_transition = nsp[2]
                
                sigma += probability_of_transition * (reward_of_transition + self.gamma * self.values[new_state])
            
            if max_value == None or max_value < sigma:
                max_value = sigma 
                max_value_action = self.actions[a]
                
        return (max_value, max_value_action)
    
    def calculate_policy(self):
        policy = np.zeros(self.number_of_states, dtype=int)
        for state in range(self.number_of_states):
            policy[state] = self.calculate_new_state_value(state)[1]
        return policy
    
    def train(self, theta):
        while True:
            max_diff = None
            for state in range(self.number_of_states):
                prev_v = self.values[state]
                
                self.values[state] = self.calculate_new_state_value(state)[0]
                
                diff_value = self.values[state] - prev_v
                if max_diff == None or diff_value > max_diff:
                    max_diff = diff_value
                
            if max_diff < theta:
                break
            
        return self.calculate_policy()

def learn_policy_using_value_iteration():
    vil = ValueIterationLearning(number_of_states=64, gamma=0.8)
    learned_policy = vil.train(theta = 0.01)
    print('Learned Policy: ')
    for s in range(learned_policy.shape[0]):
        print('State: {}, Action: {}'.format(s, learned_policy[s]))
    return learned_policy

def running_the_game_with_the_learned_policy(learned_policy):
    env = FrozenLakeEnv()
    env.reset()
    total_reward = 0
    prev_state = 0
    while True:
        env.render()
        state, reward, done, info = env.step(learned_policy[prev_state])
        prev_state = state
        total_reward += reward
        if done:
            print('Final Reward:', reward)
            print('Total Reward:', total_reward)
            break
        
    env.close()

def main():
    learned_policy = learn_policy_using_value_iteration()
    running_the_game_with_the_learned_policy(learned_policy)

if __name__ == '__main__':
    main()