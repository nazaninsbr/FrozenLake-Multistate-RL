import numpy as np 
import sys
sys.path.append("../")
from frozen_lake_env import *
import math


class nStepSarsa:
    def __init__(self, number_of_states, epsilon, alpha, gamma, n):
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
        self.n = n
        
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
        
    def calculate_discounted_return(self, all_rewards, thau, T):
        G = 0
        upper_bound = min([thau+self.n, T])
        for i in range(thau+1, upper_bound+1):
            G += math.pow(self.gamma, i-thau-1) * all_rewards[i]
        return G

    def train(self, number_of_episodes):
        for enum in range(number_of_episodes):
            episode_is_finished = False
            
            self.env.reset()
            curr_state = 0
            curr_action = self.get_action(curr_state)

            t, thau, T = -1, None, math.inf
            all_rewards, all_actions, all_states = [0], [curr_action], [0]

            while True:
                t += 1

                if t < T:
                    next_state, reward, episode_is_finished = self.move_with_action(curr_action)
                    
                    all_rewards.append(reward) 
                    all_states.append(next_state)

                    next_action = self.get_action(next_state)

                    if episode_is_finished:
                        T = t+1
                    else:
                        curr_state = next_state
                        curr_action = next_action
                        all_actions.append(curr_action)
                    
                thau = t - self.n + 1
                if thau == T-1:
                    break

                if thau >= 0:
                    G = self.calculate_discounted_return(all_rewards, thau, T)

                    if thau+self.n < T:
                        G += math.pow(self.gamma, self.n) * self.Q[all_states[thau+self.n]][all_actions[thau+self.n]]
                    
                    self.Q[all_states[thau]][all_actions[thau]] += self.alpha*(G - self.Q[all_states[thau]][all_actions[thau]])
        
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
            
    def run_one_episode(self, print_values=True):
        env = FrozenLakeEnv()
        env.reset()
        total_reward = 0
        curr_state = 0
        while True:
            if print_values:
                env.render()
            state, reward, done, info = env.step(self.get_greedy_action(curr_state))
            total_reward += reward
            curr_state = state
            if done:
                if print_values:
                    print('Reward:', reward)
                    print('Total Reward:', total_reward)
                break

        env.close()
        return total_reward
            
def main():
    for step_count in [1, 2, 3, 4, 5, 10, 20]:
        print('n =', step_count)
        sla = nStepSarsa(number_of_states = 64, epsilon=0.3, alpha=0.4, gamma=0.5, n=step_count)
        sla.train(number_of_episodes = 300)
        print('Average learned reward:', sum(sla.run_one_episode(False) for _ in range(100))/100)

if __name__ == '__main__':
    main()