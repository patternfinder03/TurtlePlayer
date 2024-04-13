# DQN Agent for TurtleTradingEnv

import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import time
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import gym_turtle_env
import gymnasium as gym
import torch.optim as optim
from collections import deque
from agents.run_single_stock import train_and_evaluate_agent_single_stock
from agents.run_multiple_stocks import run_all_stocks
from config import override_params, render_wait_time, print_state, print_rewards, train_episodes, discount_factor, exploration_decay, learning_rate


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DeepQLearningAgent:
    def __init__(self, env, state_size, action_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
        self.name = 'Deep Q-Learning Agent'
        self.batch_size = 64 # Hyperparameter

        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)  # Experience replay memory
        self.gamma = discount_factor  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = exploration_decay
        self.learning_rate = learning_rate
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # Convert to tensor
        q_values = self.model(state).detach()
        return np.argmax(q_values.cpu().numpy())  # Exploit

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                # Max over the Q-values from the next state
                target = (reward + self.gamma * torch.max(self.target_model(next_state).squeeze()).item())
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            target_f = self.model(state).squeeze()  # Get Q-values for the current state and remove extra dimension
            
            # Update the target for the chosen action
            target_f[action] = target  # Now you can directly index target_f since it's 1D
            
            # Prepare for backpropagation
            self.optimizer.zero_grad()
            loss = nn.MSELoss()(target_f, self.model(state).squeeze())  # Ensure target and predictions are aligned
            loss.backward()
            self.optimizer.step()

            # Adjust epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay


            
    def pad_state_with_ones(self, state):
        """
        Decieded to pad states with 1s on the left because don't want it to trade at the beginning
        """
        # Calculate the number of padding values needed
        pad_size = self.state_size - len(state)
        
        # Ensure pad_size is not negative
        if pad_size < 0:
            raise ValueError("State size is larger than expected. Please check the input state or configuration.")
        
        # Pad the state with 1s on the left
        padded_state = np.pad(state, (pad_size, 0), 'constant', constant_values=(1, 0))
        
        # Reshape the padded state to match the expected input shape for the network
        padded_state = np.reshape(padded_state, [1, self.state_size])
        
        return padded_state

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), name)
        
    def run_episode(self,):
        batch_size = self.batch_size
        state, info = self.env.reset()
        state = self.pad_state_with_ones(state)
        done = False
        total_reward = 0

        while not done:
            action = self.act(state)
            next_state, reward, done, truncated, info = self.env.step(action)
            next_state = self.pad_state_with_ones(next_state)

            if print_state:
                print(f"State: {next_state}")
            
            if print_rewards:
                print(f"Reward: {reward}")
            
            self.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(self.memory) > batch_size:
                self.replay(batch_size)

            if one_run:
                self.epsilon *= self.epsilon_decay
                # print(f"Exploration rate: {self.epsilon}")

        # Decay epsilon for exploration-exploitation trade-off
        if not one_run and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


        metric_details = self.env.unwrapped.get_metrics()
        return total_reward, metric_details
    
    def run_episode_w_render(self):
        batch_size = self.batch_size
        state, info = self.env.reset()
        state = self.pad_state_with_ones(state)
        done = False
        total_reward = 0

        while not done:
            action = self.act(state)
            next_state, reward, done, truncated, info = self.env.step(action)
            next_state = self.pad_state_with_ones(next_state)

            if print_state:
                print(f"State: {next_state}")
            
            if print_rewards:
                print(f"Reward: {reward}")
            
            self.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(self.memory) > batch_size:
                self.replay(batch_size)
            
            time.sleep(render_wait_time)
            
            if one_run:
                self.epsilon *= self.epsilon_decay
                # print(f"Exploration rate: {self.epsilon}")
            
            

        # Decay epsilon for exploration-exploitation trade-off
        if not one_run and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


        metric_details = self.env.unwrapped.get_metrics()
        return total_reward, metric_details

if train_episodes == 1:
    one_run = True
else:
    one_run = False



def parse_arguments():
    parser = argparse.ArgumentParser(description='Run Base Turtle Trading Strategy with specified parameters.')
    parser.add_argument('--all_stocks', dest='all_stocks', action='store_true', help='Process all stocks if this flag is set.')
    parser.add_argument('--render', dest='render', action='store_true', help='Render the environment if this flag is set.')
    parser.set_defaults(render=False, all_stocks=False)
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    
    render_mode = 'human' if args.render else None
        
    
    if args.all_stocks:
        env = gym.make('TurtleTradingEnv', render_mode=render_mode, override_params=override_params)
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        agent = DeepQLearningAgent(env, state_size, action_size)
        run_all_stocks(agent, override_params)
    else:
        env = gym.make('TurtleTradingEnv', render_mode=render_mode, override_params=override_params)
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        agent = DeepQLearningAgent(env, state_size, action_size)
        train_and_evaluate_agent_single_stock(agent, env, train_episodes)
        
# if override_params['render'] == True:
#     render_mode = 'human'
# else:
#     render_mode = None
    
# env = gym.make('TurtleTradingEnv', render_mode=render_mode, override_params=override_params)

# state_size = env.observation_space.shape[0]
# action_size = env.action_space.n

# agent = DeepQLearningAgent(env, state_size, action_size)

# train_and_evaluate_agent(agent, env, train_episodes, render_final_episode=True)

