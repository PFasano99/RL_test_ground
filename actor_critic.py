import numpy as np
import random
import tqdm
import matplotlib.pyplot as plt

#import tensorflow as tf
#import tensorflow_probability as tfp
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd

from maze_generator import *
from minosses_maze import *
from maze_ui import *
from utils import *

class actor_critic_class():
    
    def __init__(self, gamma = 0.99, epsilon = 0.0, hidden_state = 128, learning_rate = 1e-3 , load_maze = False, maze_args = [], actions= [], max_episodes = 1000, num_iterations = 1000, player_char = 9):
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = actions
        self.episodes = max_episodes
        self.num_iteration = num_iterations
        self.player_char = player_char

        self.hidden_size = hidden_state
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
          
        self.actor_network = Actor(len(path[0]), len(actions))
        self.actor_opt = torch.optim.Adam(self.actor_network.parameters(),self.learning_rate)
        
        self.critic_network = Critic(len(path[0]), 1)
        self.critic_opt = torch.optim.Adam(self.critic_network.parameters(),self.learning_rate)
        
        self.device = "cpu"
        
    

    def a2c(self):
        policy = np.zeros([len(self.path),len(self.actions)])
        all_rewards = []
        for i in tqdm(range(0, self.episodes), desc="A2C-episodes"):
                
            utils = common_functions(self.maze)
            state = self.start_coord
            r = 0
            steps = 0
            done = False
            
            while done == False and steps < self.num_iteration:               
                steps +=1
                # Choose an action using an epsilon-greedy policy based on the Q-network
                action, action_id = self.epsilon_greedy_policy(state, self.epsilon)
                # Take the sampled action
                next_state, reward = utils.action_value_function(state, action, self.finish_coord)
                if reward >= 0: done = True  

                pol_step, action_probs = self.extract_policy(state)
                policy[utils.find_index_of_coordinate(self.path, state), np.argmax(pol_step)] = 1 
                
                # Update Critic
                state_value = self.critic_network(torch.FloatTensor(state), torch.FloatTensor(action_id))
                action2, action_id2 = self.epsilon_greedy_policy(next_state, self.epsilon)
                next_state_value = self.critic_network(torch.FloatTensor(next_state), torch.FloatTensor(action_id2))
                target = reward + self.gamma * next_state_value.item()
                critic_loss = nn.MSELoss()(state_value, torch.FloatTensor([target]))
                self.critic_opt.zero_grad()
                critic_loss.backward()
                self.critic_opt.step()

                # Update Actor
                advantage = target - state_value.item()
                actor_loss = -torch.log(action_probs[action_id]) * advantage ##check
                self.actor_opt.zero_grad()
                actor_loss.backward()
                self.actor_opt.step()

                state = next_state
                
                r+=reward
                
            all_rewards.append(r)
        
        best_r = -1e10
        best_r_g = 0
        g = 0
        for rew in all_rewards:
            print("Episode: ", g, " rewards: ",rew)
            if rew > best_r: 
                best_r = rew
                best_r_g = g
            g+=1
        
        print("Best reward: ", best_r, " on episode ", best_r_g)
                
       # print(policy)
        return policy
    
    def epsilon_greedy_policy(self, state, epsilon):
            if random.uniform(0,1) < epsilon:
                action = random.randrange(0,len(self.actions))
                return self.actions[action], action
            else:            
                q_values = self.actor_network(torch.FloatTensor(state))
                action = torch.argmax(q_values).item()
        
                return self.actions[action], action       

    # Evaluate the actor network on a state to get the action probabilities
    def extract_policy(self, state):
        state_tensor = torch.FloatTensor(state)
        action_probs = self.actor_network(state_tensor)
        policy = action_probs.detach().numpy()
        return policy, action_probs
    
    
    def choose_action(self, observation):
        state = torch.tensor([observation], dtype=torch.float).to(self.device)
        probabilities = self.actor_network.forward(state) 
        probabilities = F.softmax(probabilities, dim=1)
        action_probs = torch.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_prob = action_probs.log_prob(action)
        self.log_prob = log_prob
        
        return action.item()

    def learn(self, state, reward, state_, done):
        self.actor_opt.zero_grad()

        state = torch.tensor([state], dtype=torch.float).to(self.device)
        state_ = torch.tensor([state_], dtype=torch.float).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.device)

        critic_value = self.critic_network.forward(state)
        critic_value_ = self.critic_network.forward(state_)

        delta = reward + self.gamma*critic_value_*(1-int(done)) - critic_value

        actor_loss = -self.log_prob*delta
        critic_loss = delta**2

        (actor_loss + critic_loss).backward()
        self.actor_opt.step()

    def a2c_sol(self):
        policy = np.zeros([len(self.path),len(self.actions)])
        all_rewards = []
        for i in tqdm(range(0, self.episodes), desc="A2C-episodes"):
                
            utils = common_functions(self.maze)
            state = self.start_coord
            r = 0
            steps = 0
            done = False
            
            while done == False and steps < self.num_iteration:               
                steps +=1
                # Choose an action using an epsilon-greedy policy based on the Q-network
                #action, action_id = self.epsilon_greedy_policy(state, self.epsilon)
                action = self.choose_action(state)
                # Take the sampled action
                next_state, reward = utils.action_value_function(state, self.actions[action], self.finish_coord)
                if reward > 0: done = True
                #observation_, reward, done, info = env.step(action)
                r += reward
                self.learn(state, reward, next_state, done)
                state = next_state
                
            
            all_rewards.append(r)
        
        best_r = -1e10
        best_r_g = 0
        g = 0
        for rew in all_rewards:
            print("Episode: ", g, " rewards: ",rew)
            if rew > best_r: 
                best_r = rew
                best_r_g = g
            g+=1
        
        print("Best reward: ", best_r, " on episode ", best_r_g)
            
            

    

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
    
class Critic(nn.Module):
    
    def __init__(self, input_size, output_size):
        super(Critic, self).__init__()
        
        self.fc_state = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.Linear(512,256)
        )
        
        self.fc_actions = nn.Sequential(
            nn.Linear(output_size, 256),
            nn.ReLU()
            )
        
        self.fc = nn.Sequential(
            #nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(256,1)
        )
       
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
       
    def forward(self, state):#, action):
        st = self.fc_state(state)
        #ac = self.fc_actions(action)
        #x = torch.cat((st,ac),dim=1)
        x = self.fc(st)
        
        return x

class Actor(nn.Module):
    
    def __init__(self, input_size, output_size ):
        super(Actor, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
            nn.Tanh()
        )
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, state):
        x = self.fc(state)
        return x
    

actions = ["up","down","right","left","jump_up","jump_down","jump_right","jump_left"]
utils = common_functions()
path_to_file_maze = "./saved_maze/maze5"

actor_critic_class(gamma=0.99, load_maze=True, maze_args=[path_to_file_maze], actions=actions).a2c_sol()
input()