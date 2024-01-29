import numpy as np
import random
import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from maze_generator import *
from minosses_maze import *
from maze_ui import *
from utils import *

class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class DQNAgent:
    def __init__(self,  output_size, maze_path, replay_buffer_size=50000, cuda = True, gamma=0.99, batch_size = 1024, lr=0.001, epsilon_start=1, epsilon_decay=0.996, epsilon_min=0.01):
        generator = maze_generator()      
        
        maze, path, start_coord, finish_coord = generator.load_maze_from_csv(maze_path)
        self.maze = maze
        self.path = path
        self.size_x = len(maze)
        self.size_y = len(maze[0])
        self.start_coord = start_coord
        self.finish_coord = finish_coord
    
        self.device = "cpu"
        if torch.cuda.is_available() and cuda:
            self.device = "cuda" 
        print(self.device)
        
        self.input_size = len(self.path[0])
        self.output_size = output_size
        self.q_network = QNetwork(self.input_size, output_size).to(self.device)
        self.target_network = QNetwork(self.input_size, output_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_function = nn.MSELoss()
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.replay_buffer = replay_buffer(capacity=replay_buffer_size, buffer=[])
        self.batch_size = batch_size
        self.target_network_frequency = 5

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.output_size)
        else:
            with torch.no_grad():
                q_values = self.q_network(torch.tensor(state, dtype=torch.float32).to(self.device))
                return torch.argmax(q_values).item()

    def train(self):
        if self.replay_buffer.__len__() > self.batch_size:
            # we sample from our buffer
            states, actions, rewards, next_states, dones = self.replay_buffer.sample_batch(self.batch_size)
            states = torch.tensor(np.asarray(states), dtype=torch.float32).to(self.device)
            actions = torch.tensor(np.asarray(actions), dtype=torch.int64).to(self.device)
            next_states = torch.tensor(np.asarray(next_states), dtype=torch.float32).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            dones = torch.FloatTensor(dones).to(self.device)
            
            next_q_values = self.target_network(next_states)
            q_values = self.q_network(states).gather(1, actions.unsqueeze(-1)).squeeze()         
            target = rewards + (1 - dones) * self.gamma * next_q_values.max(1)[0]
            
            loss = self.loss_function(q_values, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def update_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def update_target_network(self):
        # update target network
        for target_network_param, q_network_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_network_param.data.copy_(
                self.gamma * q_network_param.data + (1.0 - self.gamma) * target_network_param.data
            )

class replay_buffer():
    def __init__(self, capacity, buffer):
        self.capacity = capacity
        self.buffer = buffer

    def add_experience(self, state, action, reward, next_state, done):
        # we overwrite the oldest examples with the newest
        if len(self.buffer) >= self.capacity:
            self.buffer.pop()
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def update_buffer_at(self, index, new_buffer):
        self.buffer[index] = new_buffer

    def sample_batch(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

def train_dqn(episodes=1500, path_to_file_maze = "./saved_maze/maze4", batch_size = 512, cuda = True, epsilon_start = 1, epsilon_decay = 0.999, replay_buffer_size=50000):
    
    actions = ["up","down","right","left"]#,"jump_up","jump_down","jump_right","jump_left"]

    agent = DQNAgent(output_size=len(actions), maze_path=path_to_file_maze, batch_size=batch_size, cuda=cuda, epsilon_start=epsilon_start, epsilon_decay=epsilon_decay, replay_buffer_size=replay_buffer_size)
    utils = common_functions(agent.maze)
    all_rewards = []

    print(f"\nstarting state = {agent.start_coord} Final state = {agent.finish_coord} \n")

    for episode in tqdm(range(episodes), desc="Running DQN episodes"):
        state = agent.start_coord 
        total_reward = len(agent.path)*0.4
        done = False
        steps = 0
        control_asns = {}

        while True:
            steps+=1
            action_id = agent.select_action(state)
            action = actions[action_id]
            next_state, reward = utils.action_value_function(state, action, agent.finish_coord)
            
            key = str(action) + str(state[0]) + str(state[1]) + str(next_state[0]) + str(next_state[1])

            # Initialize the dictionary if not already present
            if key not in control_asns:
                control_asns[key] = 0   

            # Increment the value associated with the key
            control_asns[key] += 1

            if next_state == agent.finish_coord: 
                done = True

            agent.replay_buffer.add_experience(state, action_id, reward, next_state, done)

            agent.train()

            state = next_state
            total_reward += reward

            # if we are stuck in a loop it choose a completely random action
            if (control_asns[key] >= 20):
                
                new_action_id = action_id
                while new_action_id == action_id:
                    new_action_id = np.random.choice(len(actions))

                action = actions[new_action_id]
                next_state, reward = utils.action_value_function(state, action, agent.finish_coord)
                if next_state == agent.finish_coord: 
                    done = True

                agent.replay_buffer.add_experience(state, new_action_id, reward, next_state, done)

                key = str(action) + str(state[0]) + str(state[1]) + str(next_state[0]) + str(next_state[1])

                # Initialize the dictionary if not already present
                if key not in control_asns:
                    control_asns[key] = 0   

                # Increment the value associated with the key
                control_asns[key] += 1

                state = next_state
                total_reward += reward


            if (control_asns[key] >= 50 and total_reward < 0) or done or total_reward < -1000: 
                break


        if episode % agent.target_network_frequency == 0 or done:
            print("updating target network")
            agent.update_target_network()

        if episode % 50 == 0:
            print("agent.epsilon ",agent.epsilon)

        all_rewards.append(total_reward)
        agent.update_epsilon()
        print(f'Episode {episode}: Start reward: {len(agent.path)*0.4} Total reward = {total_reward} \n Done: {done} Final state: {state} ')
     

    print("agent.epsilon ",agent.epsilon)
    print("max reward is: ", np.max(all_rewards), " at ", (np.argmax(all_rewards)))
    
    return np.max(all_rewards)


#train_dqn()
