import numpy as np
import random
import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


from maze_generator import *
from minosses_maze import *
from maze_ui import *
from utils import *

class deep_Q_learning_class():
    
    def __init__(self, gamma = 0.99, epsilon = 0.2, load_maze = False, maze_args = [], actions= [], num_iterations = 3000, episodes = 100, batch_size = 32, player_char = 9):
        self.gamma = gamma
        self.actions = actions
        self.num_iteration = num_iterations
        self.episodes = episodes
        self.player_char = player_char
        self.epsilon = epsilon

        generator = maze_generator()      
        
        if load_maze:
            path = maze_args[0]
            maze, path, start_coord, finish_coord = generator.load_maze_from_csv(path)
            self.maze = maze
            self.path = path
            self.size_x = len(maze)
            self.size_y = len(maze[0])
            self.start_coord = start_coord
            self.finish_coord = finish_coord
        else:
            self.size_x = maze_args[0]
            self.size_y = maze_args[1]
            self.start_coord = maze_args[2]
            self.finish_coord = maze_args[3]
            maze, path = generator.generate_maze(size_x=maze_args[0], size_y =maze_args[1], start_coord = maze_args[2], finish_coord = maze_args[3], n_of_turns = maze_args[4], log = False)
            self.maze = maze
            self.path = path

        self.deepQ_network = deep_Q_network(len(self.path[0]), len(self.actions))
        self.target_deepQ_network = target_Q_network(len(self.path[0]), len(self.actions))
        self.replay_buffer = replay_buffer(10000, [])
        self.batch_size = batch_size

    def epsilon_greedy_policy(self, state, epsilon):

        utils = common_functions()
        if random.uniform(0,1) < epsilon:
            action = random.randrange(0,len(self.actions))
            return self.actions[action], action
        else:            
            q_values = self.deepQ_network(torch.FloatTensor(state))
            action = torch.argmax(q_values).item()
    
            return self.actions[action], action

    def build_policy_from_q(self, q, state):
        policy = np.zeros([len(self.path),len(self.actions)])
        values = F.softmax(q, dim=1).detach().numpy()


        for i in range(0,len(values)):
            best_action = np.argmax(values[i])
            policy[i][best_action] = 1 
        
        return policy
    

    def DQN_agent(self):

        target_update_frequency = 2  # how often to update target Q-network

        policy = np.zeros([len(self.path),len(self.actions)])

        for i in tqdm(range(0, self.episodes), desc="DQN-episodes"):
            
            utils = common_functions(self.maze)
            r = 0  
            state = self.start_coord
            done = False

            steps = 0
            while done == False and steps < self.num_iteration:
                #print("step: ",steps, " reached final state: ", done)
                steps+=1
                
                # Choose an action using an epsilon-greedy policy based on the Q-network
                action, action_id = self.epsilon_greedy_policy(state, self.epsilon)
                # Take the sampled action
                next_state, reward = utils.action_value_function(state, action, self.finish_coord)
                if reward >= 0: done = True
                # Store (state, action, reward, next_state) in the replay buffer
                new_experience = (state, action_id, reward, next_state, done)
                self.replay_buffer.add_experience(new_experience)

                # Sample a mini-batch from the replay buffer 
                states, actions, rewards, next_states, dones = self.replay_buffer.sample_batch(self.batch_size)

                # Calculate target Q-values using the target Q-network
                with torch.no_grad():
                    target_q_values_next = self.target_deepQ_network(torch.FloatTensor(next_states))
                    target_q_values_next_max, _ = target_q_values_next.max(dim=1)
                    target_q_values = torch.FloatTensor(rewards) + self.gamma * (1 - torch.FloatTensor(dones)) * target_q_values_next_max

                # Update the Q-network using the loss between predicted Q-values and target Q-values
                q_values = self.deepQ_network(torch.FloatTensor(states))
                q_values_actions = q_values[range(len(actions)), torch.LongTensor(actions)]
                loss = nn.MSELoss()(q_values_actions, target_q_values)
                optimizer = optim.Adam(self.deepQ_network.parameters(), lr=0.001)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update the target Q-network weights periodically
                if i % target_update_frequency == 0:
                    self.target_deepQ_network.load_state_dict(self.deepQ_network.state_dict())

                # Update the state
                state = next_state
                # Update the total episode reward
                r+=reward
                
                s = common_functions().find_index_of_coordinate(self.path, state)
                values = F.softmax(q_values, dim=1).detach().numpy()[0]
                best_action = np.argmax(values)
                policy[s][best_action] = 1

            # Extract the policy from the Q-network using softmax
            #policy = self.build_policy_from_q(q_values)

        return q_values, policy


class deep_Q_network(nn.Module):
    """
        - input_size would be the size of the state vector
        - output_size would be the number of possible actions
    """
    def __init__(self, input_size, output_size):
        super(deep_Q_network, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class target_Q_network(nn.Module):
    def __init__(self, input_size, output_size):
        super(target_Q_network, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class replay_buffer():
    def __init__(self, capacity, buffer):
        self.capacity = capacity
        self.buffer = buffer

    def add_experience(self, experience):
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        #else: print("buffer is full")

    def update_buffer_at(self, index, new_buffer):
        self.buffer[index] = new_buffer

    def sample_batch(self, batch_size):
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
    

actions = ["up","down","right","left","jump_up","jump_down","jump_right","jump_left"]
utils = common_functions()
path_to_file_maze = "./saved_maze/maze5"

batch_sizes = [32, 64,128]
for batch_size in batch_sizes:
    q, policy = deep_Q_learning_class(gamma=0.99, epsilon=0.2, batch_size = batch_size, load_maze=True, maze_args=[path_to_file_maze], actions=actions).DQN_agent()
    Q_learning_score = utils.play_episode(policy, path_to_file_maze)
