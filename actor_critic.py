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

# Define the experience buffer class
class replay_buffer():
    def __init__(self, capacity, buffer):
        self.capacity = capacity
        self.buffer = buffer

    def add_experience(self, state, action, reward, next_state, done):
        # we overwrite the oldest examples with the newest
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)  # Remove the oldest experience
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample_batch(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

class actor_critic_class():

    def __init__(self, gamma = 0.99, learning_rate = 3e-4 , load_maze = True, maze_args = [], actions= [], max_episodes = 1500, player_char = 9, epsilon_start=1, epsilon_decay=0.998, epsilon_min=0.01):
        self.gamma = gamma
        self.actions = actions
        self.episodes = max_episodes
        self.player_char = player_char

        self.learning_rate = learning_rate  
            
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

        self.num_episodes = max_episodes
        self.num_outputs = len(self.actions)
        self.epsilon_start = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda:1" 
        print(self.device)

        self.action_size = len(self.actions)
        self.state_size = len(self.path[0])

        # Initialize the ActorCritic network
        self.agent = ActorCritic(self.state_size, self.action_size)

        # Define the optimizer
        self.optimizer = optim.Adam(self.agent.parameters(), lr=learning_rate)

        self.replay_buffer = replay_buffer(capacity=20000, buffer=[])

    def update(self):
        num_episodes = self.num_episodes
        all_rewards = []
        utils = common_functions(self.maze)
        print(f"\nstarting state = {self.start_coord} Final state = {self.finish_coord} \n")
        # Define the training loop
        for episode in tqdm(range(num_episodes), desc="Running a2c episodes"):
            # Initialize the environment
            state = self.start_coord
            done = False
            total_reward = len(self.path)*0.4
            epsilon = max((self.epsilon_start * self.epsilon_decay**episode), self.epsilon_min)
            control_asns = {}

            while True:
                if np.random.rand() < epsilon:
                    action_id = np.random.choice(self.action_size)
                    probs = np.zeros(self.action_size)
                    probs[action_id] = 1/self.action_size
                    probs = torch.tensor(probs, dtype=torch.float32)
                else:
                    probs, _ = self.agent(torch.tensor(state, dtype=torch.float32))
                    action_id = np.argmax(probs.detach().numpy())
                # Take a step in the environment
                next_state, reward = utils.action_value_function(state, self.actions[action_id], self.finish_coord)
                
                key = str(action_id) + str(state[0]) + str(state[1]) + str(next_state[0]) + str(next_state[1])
                # Initialize the dictionary if not already present
                if key not in control_asns:
                    control_asns[key] = 0   
                # Increment the value associated with the key
                control_asns[key] += 1

                if next_state == self.finish_coord: 
                    done = True

                total_reward += reward

                # Calculate the TD error and loss
                _, next_val = self.agent(torch.tensor(next_state, dtype=torch.float32))
                td_target = reward + self.gamma * next_val * (1 - done)
                td_error = td_target - self.agent(torch.tensor(state, dtype=torch.float32))[1]
                actor_loss = -torch.log(probs.squeeze(0)[action_id]) * td_error.detach()
                critic_loss = torch.square(td_error)
                loss = actor_loss + critic_loss

                # Update the network
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                

                # Set the state to the next state
                state = next_state

                # if we are stuck in a loop it choose a completely random action
                if (control_asns[key] >= 20):
                    
                    new_action_id = action_id
                    while new_action_id == action_id:
                        new_action_id = np.random.choice(len(self.actions))

                    action = actions[new_action_id]
                    next_state, reward = utils.action_value_function(state, action, self.finish_coord)
                    if next_state == self.finish_coord: 
                        done = True

                    key = str(action) + str(state[0]) + str(state[1]) + str(next_state[0]) + str(next_state[1])

                    # Initialize the dictionary if not already present
                    if key not in control_asns:
                        control_asns[key] = 0   

                    # Increment the value associated with the key
                    control_asns[key] += 1

                    state = next_state
                    total_reward += reward

                if (control_asns[key] >= 50 and total_reward < 0) or done:
                    #print(f"init done {done} \n with {next_state} == {self.finish_coord}")
                    break

            if episode % 50 == 0:
                print("epsilon = ", epsilon)
            
            # Print the total reward for the episode
            print(f'Episode {episode}: Total reward = {total_reward} \n Done: {done} Final state: {state} ')
            all_rewards.append(total_reward)
        
        print("fnal epsilon = ", epsilon)
        print("max reward is: ", np.max(all_rewards), " at ", (np.argmax(all_rewards)))

# Define the actor-critic network
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc_pi = nn.Linear(256, action_dim)
        self.fc_v = nn.Linear(256, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        pi = torch.softmax(self.fc_pi(x), dim=-1)
        v = self.fc_v(x)
        
        return pi, v
    

path_to_file_maze = "./saved_maze/maze4"
actions = ["up","down","right","left"]#,"jump_up","jump_down","jump_right","jump_left"]

agent = actor_critic_class(maze_args=[path_to_file_maze], actions = actions).update()
