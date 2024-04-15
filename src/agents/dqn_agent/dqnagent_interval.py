import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from config import print_rewards, print_state, render_episode, render_wait_time


class DQNAgentInveral:
    def __init__(self, env, learning_rate=0.001, discount_factor=0.95, exploration_rate=1.0, exploration_decay=0.995, min_exploration_rate=0.01, memory_size=512, batch_size=64):
        self.learning_rate_original = learning_rate
        self.discount_factor_original = discount_factor
        self.exploration_rate_original = exploration_rate
        self.exploration_decay_original = exploration_decay
        self.min_exploration_rate_original = min_exploration_rate
        self.memory_size_original = memory_size    
        
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.name = 'Deep Q-Learning Agent Interval'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model().to(self.device)
        
        self.exploration_rates = [1]
        self.reward_intervals = []
        
        self.run_time = 0
        self.max_run_time = 252
        
    def reset_params(self):
        self.learning_rate = self.learning_rate_original
        self.discount_factor = self.discount_factor_original
        self.exploration_rate = self.exploration_rate_original
        self.exploration_decay = self.exploration_decay_original
        self.min_exploration_rate = self.min_exploration_rate_original
        self.exploration_rates = [1]
        self.reward_intervals = []
        self.memory = deque(maxlen=self.memory_size_original)
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        class NeuralNet(nn.Module):
            def __init__(self, input_size, output_size):
                super(NeuralNet, self).__init__()
                self.fc1 = nn.Linear(input_size, 24)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(24, 24)
                self.fc3 = nn.Linear(24, output_size)

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                x = self.fc3(x)
                return x

        state_shape = self.env.observation_space.shape[0]
        action_size = self.env.action_space.n
        model = NeuralNet(state_shape, action_size)
        return model


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)  # Add batch dimension
        if np.random.rand() <= self.exploration_rate:
            return random.randrange(self.env.action_space.n)
        self.model.eval()
        with torch.no_grad():
            act_values = self.model(state)
        self.model.train()
        return act_values.max(1)[1].item()  # returns action

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*minibatch)
        
        state_batch = torch.tensor(state_batch, dtype=torch.float32).to(self.device)
        action_batch = torch.tensor(action_batch, dtype=torch.long).to(self.device)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32).to(self.device)
        next_state_batch = torch.tensor(next_state_batch, dtype=torch.float32).to(self.device)
        done_batch = torch.tensor(done_batch, dtype=torch.float32).to(self.device)

        # Get predicted Q-values for next states from the model
        Q_targets_next = self.model(next_state_batch).detach().max(1)[0]
        # Compute Q targets for current states
        Q_targets = reward_batch + (self.discount_factor * Q_targets_next * (1 - done_batch))

        # Get expected Q values from the model
        Q_expected = self.model(state_batch).gather(1, action_batch.unsqueeze(-1)).squeeze(-1)

        # Compute loss
        loss = nn.MSELoss()(Q_expected, Q_targets)
        
        # Optimize the model
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if self.exploration_rate > self.min_exploration_rate:
        #     self.exploration_rate *= self.exploration_decay

    def run_episode(self):
        state, info = self.env.reset()
        done, truncated = False, False
        total_reward = 0
        
        days = []
        state_memory = []
        next_state_memory = []
        pre_prices = []
        previous_segment_prices = []
        segment_prices = []
        agent_windows = []
        agent_actions_converted = []
        agent_actions = []
        first_exit = True
        starting_index = 0
        
        
        reward_iter = 0
        

        while not done and not truncated:
            if render_episode:
                time.sleep(render_wait_time)
             
            self.exploration_rates.append(self.exploration_rate)   
            days.append(starting_index)
            state_memory.append(state)        
            action = self.get_action(state)
            # action = 2
            
            high_price = self.env.unwrapped.controller.trader.high_price_list[-1]
            low_price = self.env.unwrapped.controller.trader.low_price_list[-1]
            close_price = self.env.unwrapped.controller.trader.close_price_list[-1]
            segment_prices.append({"High": high_price, "Low": low_price, "Close": close_price})
            
                
            next_state, reward, done, truncated, info = self.env.step(action)
            agent_windows.append(self.env.unwrapped.controller.trader.current_period)   
            agent_action = self.env.unwrapped.controller.trader.action_list[-1]
            agent_actions.append(agent_action)
            if agent_action == 0:
                agent_actions_converted.append("Increase")
            elif agent_action == 1:
                agent_actions_converted.append("Decrease")
            else:
                agent_actions_converted.append("Nothing") 
                
            next_state_memory.append(next_state)
            if info['is_close_X']:
                if not first_exit:
                    pre_prices = previous_segment_prices[-self.env.unwrapped.controller.trader.absolute_max:]
                    
                rewards, solver_data = self.env.unwrapped.controller.get_interval_reward_X(pre_prices=pre_prices,
                                stock_prices=segment_prices, trader_actions=agent_actions_converted, trader_windows=agent_windows
                                , first_exit=first_exit)
                
                self.reward_intervals.extend(rewards)
                
                for day_t, reward_t, state_t, next_state_t, action_t, agent_window_t, optimal_t in zip(days, rewards, state_memory, next_state_memory, agent_actions, agent_windows, solver_data):
                    if print_state:
                        print(f"Day: {day_t}, State: {state_t}")
                    if print_rewards:
                        print(f"Day:{day_t:4},Rew:{reward_t:5.2f},Act:{action_t:1},Per:{agent_window_t:2},ExpRate:{round(self.exploration_rate, 2):4},RIt:{reward_iter:3},IL:{int(optimal_t['smoothed_ideal'])},MIN:{optimal_t['min']},MAX:{optimal_t['max']},OA:{optimal_t['optimal_action']:13},EP:{optimal_t['exit_period']}")
                    self.remember(state_t, action_t, reward_t, next_state_t, False)
                    # self.replay()
                    total_reward += reward_t
                    
                self.replay()
                    
                
                reward_iter += 1    
                previous_segment_prices.extend(segment_prices)
                
                days = []
                segment_prices = []
                agent_windows = []
                agent_actions = []
                agent_actions_converted = []
                state_memory = []
                next_state_memory = []
                first_exit = False

            state = next_state
            
            self.exploration_rate *= self.exploration_decay
            if self.exploration_rate < 0.1:
                self.run_time += 1
                if self.run_time >= self.max_run_time:
                    self.run_time = 0
                    self.exploration_rate = 1
                else:
                    self.exploration_rate = 0
                    
            starting_index += 1
            

        else:
            self.exploration_rate *= self.exploration_decay
        metric_details = self.env.unwrapped.get_metrics()
        print("Exploration Decay: ", self.exploration_decay)
        
        for _ in range(0, len(segment_prices)):
            self.reward_intervals.append(0)
        
        return total_reward, metric_details
    


