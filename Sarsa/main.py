import numpy as np 
import sys
sys.path.append("../")
from frozen_lake_env import *
import random 

class SarsaLearningAgent:
    def __init__(self, number_of_states, epsilon, alpha, gamma):
        self.env = FrozenLakeEnv()
        # learning parameters
        self.actions = {'LEFT':0, 'RIGHT':2, 'UP':3, 'DOWN':1}
        self.n_actions = len(self.actions.keys())
        self.number_of_states = number_of_states
        # soft policy variables
        self.Q = np.zeros((number_of_states, len(self.actions.keys())))
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        
    def move_with_action(self, action):
        state, reward, done, info = self.env.step(action)
        return state, reward, done
    
    def get_action(self, state):
        if np.random.random()>self.epsilon:
            return np.argmax(self.Q[state])
        else:
            return np.random.randint(self.n_actions)
    
    def decrease_epsilon(self):
        self.epsilon = self.epsilon*0.9

    def print_best_actions_per_state(self):
        for state in range(len(self.Q)):
            print('State: {}, Action: {}'.format(state, self.get_greedy_action(state)))
    
    def update_Q_value(self, state, action, reward, next_state_q_value):
        self.Q[state][action] = self.Q[state][action] + self.alpha * (reward + (self.gamma * next_state_q_value) - self.Q[state][action])
        
    def train(self, number_of_episodes):
        for _ in range(number_of_episodes):
            episode_is_finished = False
            
            self.env.reset()
            curr_state = 0
            curr_action = self.get_action(curr_state)
            
            while not episode_is_finished:
                next_state, reward, episode_is_finished = self.move_with_action(curr_action)
                
                next_action = self.get_action(next_state)
                next_state_q_value = self.Q[next_state][next_action]
                
                self.update_Q_value(curr_state, curr_action, reward, next_state_q_value)
                
                curr_state = next_state
                curr_action = next_action
        
            self.decrease_epsilon()
    
    def get_greedy_action(self, state):
        max_Q, max_action = None, None
        for action in self.actions.keys():
            a = self.actions[action]
            q = self.Q[state][a] 
            if max_Q == None or q>max_Q:
                max_Q = q
                max_action = a
        return max_action
            
    def run_one_episode(self):
        env = FrozenLakeEnv()
        env.reset()
        total_reward = 0
        curr_state = 0
        while True:
            env.render()
            state, reward, done, info = env.step(self.get_greedy_action(curr_state))
            total_reward += reward
            curr_state = state
            if done:
                print('Reward:', reward)
                print('Total Reward:', total_reward)
                break

        env.close()
            
def main():
    sla = SarsaLearningAgent(number_of_states = 64, epsilon=0.4, alpha=0.5, gamma=0.5)
    sla.train(number_of_episodes = 1000)
    sla.run_one_episode()

if __name__ == '__main__':
    main()