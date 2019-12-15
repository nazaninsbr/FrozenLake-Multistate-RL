import numpy as np 
import sys
sys.path.append("../")
from frozen_lake_env import *

class MonteCarloLearning:
    def __init__(self, number_of_states, gamma, epsilon):
        self.env = FrozenLakeEnv()
        self.reset_env()
        # learning parameters
        self.actions = {'LEFT':0, 'RIGHT':2, 'UP':3, 'DOWN':1}
        self.number_of_states = number_of_states
        self.gamma = gamma
        self.epsilon = epsilon

        self.policy = np.zeros((self.number_of_states, len(self.actions.keys())))
        self.Q = np.zeros((self.number_of_states, len(self.actions.keys())))
        self.Returns = [[[] for j in range(len(self.actions.keys()))] for i in range(self.number_of_states)] 
    
    def reset_env(self):
        self.env.reset()
        
    def get_action(self, state):
        if np.random.random()>self.epsilon:
            return np.argmax(self.Q[state])
        else:
            return np.random.randint(len(self.actions.keys()))
    
    def generate_an_episode(self):
        episode = []
        self.reset_env()
        prev_state = 0
        while True:
            action = self.get_action(prev_state)
            state, reward, done, info = self.env.step(action)
            episode.append({'state':prev_state, 'action':action, 'reward':reward})
            prev_state = state
            if done:
                break
        return episode

    def is_this_the_states_first_visit(self, episode, this_step):
        this_state_action = (episode[this_step]['state'], episode[this_step]['action'])
        all_state_actions_up_to_here = [(episode[i]['state'], episode[i]['action']) for i in range(this_step)]
        if this_state_action in all_state_actions_up_to_here:
            return False
        return True
    
    def update_policy(self):
        for state in range(self.number_of_states):
            max_action = np.argmax(self.Q[state])
            for action in range(len(self.actions.keys())):
                if action == max_action:
                    self.policy[state][action] = 1 - self.epsilon + self.epsilon/len(self.actions.keys())
                else:
                    self.policy[state][action] = self.epsilon/len(self.actions.keys())
    
    def calculate_policy(self):
        policy = np.zeros(self.number_of_states, dtype=int)
        for state in range(self.number_of_states):
            policy[state] = np.argmax(self.Q[state])
        return policy

    def run_the_game_with_the_learned_policy(self, learned_policy):
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

    def train(self, number_of_episodes):
        for enum in range(number_of_episodes):
            episode = self.generate_an_episode()
            print('({}) Episode Length: {}'.format(enum, len(episode)))
            T = len(episode) - 1
            G = 0
            for step_t in range(len(episode)):  
                this_step = T - step_t
                G = self.gamma * G + episode[this_step]['reward']

                if self.is_this_the_states_first_visit(episode, this_step):
                    self.Returns[episode[this_step]['state']][episode[this_step]['action']].append(G)
                    self.Q[episode[this_step]['state']][episode[this_step]['action']] = sum(self.Returns[episode[this_step]['state']][episode[this_step]['action']])/len(self.Returns[episode[this_step]['state']][episode[this_step]['action']])

            self.update_policy()
        return self.calculate_policy()

def main():
    mc = MonteCarloLearning(number_of_states=64, gamma=0.8, epsilon=0.1)
    learned_policy = mc.train(200)
    mc.run_the_game_with_the_learned_policy(learned_policy)

if __name__ == '__main__':
    main()