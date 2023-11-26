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
    
    def __init__(self, gamma = 0.9, epsilon = 0.2, load_maze = False, maze_args = [], actions= [], num_iterations = 500, episodes = 2000, batch_size = 64, player_char = 9):
        self.gamma = gamma
        self.actions = actions
        self.num_iteration = num_iterations
        self.episodes = episodes
        self.player_char = player_char
        self.epsilon = epsilon
        
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"

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
        self.optimizer = torch.optim.Adam(self.deepQ_network.parameters())
        self.optimizer_tsq = torch.optim.Adam(self.target_deepQ_network.parameters())
        self.MSE_loss = nn.MSELoss()
        
    def epsilon_greedy_policy(self, state, epsilon):

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

        target_update_frequency = 10  # how often to update target Q-network

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
                    target_q_values = (torch.FloatTensor(rewards) + self.gamma * (1-torch.FloatTensor(dones)) * target_q_values_next_max)

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

    def vanillaDQN(self):
        policy = np.zeros([len(self.path),len(self.actions)])
        episode_rewards = []
        best_reward = -1e12
        loss = []
        
        for i in tqdm(range(0, self.episodes), desc="DQN-episodes"):
            
            utils = common_functions(self.maze)
            r = 0  
            state = self.start_coord
            done = False

            steps = 0
            while done == False and steps < self.num_iteration:
                #print("step: ",steps, " reached final state: ", done)
                steps+=1
                action, action_id = self.epsilon_greedy_policy(state, self.epsilon)
                next_state, reward = utils.action_value_function(state, action, self.finish_coord)
                if reward >= 0: done = True
                r+=reward
                
                # Store (state, action, reward, next_state) in the replay buffer
                self.replay_buffer.add_experience(state, action_id, reward, next_state, done)
                
                
                if len(self.replay_buffer.buffer) > self.batch_size:
                    l = self.update() 
                    loss.append(l)
                if done: break
                
                state = next_state

            if best_reward < r:
                best_reward = r
            episode_rewards.append(r)
            
        self.plot_loss(loss)
            
        for ep in range(0,len(episode_rewards)):  
            print("epsiode: ",ep," reward: ", episode_rewards[ep])
            
        print("best-reward: ", best_reward)
            
    def compute_loss(self, batch):
        states, actions, rewards, next_states, dones = batch
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones)

        curr_Q = self.deepQ_network.forward(states).gather(1, actions.unsqueeze(1))
        curr_Q = curr_Q.squeeze(1)
        next_Q = self.deepQ_network.forward(next_states)
        max_next_Q = torch.max(next_Q, 1)[0]
        expected_Q = rewards.squeeze(1) + self.gamma * max_next_Q

        loss = self.MSE_loss(curr_Q, expected_Q)
        return loss

    def update(self):
        batch = self.replay_buffer.sample_batch(self.batch_size)
        loss = self.compute_loss(batch)
        loss_copy = loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss_copy

    def plot_loss(loss = []):
        
        # Generate data for plotting
        x_values = np.linspace(0, len(loss)+4, len(loss))
        y_values = loss

        # Plot the loss function
        plt.plot(x_values, y_values, label='Loss Function')
        plt.title('Loss Function Visualization')
        plt.xlabel('Parameter Values')
        plt.ylabel('Loss Value')
        plt.legend()
        plt.grid(True)
        plt.show()
                

class deep_Q_network(nn.Module):
    """
        - input_size would be the size of the state vector
        - output_size would be the number of possible actions
    """
    def __init__(self, input_size, output_size):
        super(deep_Q_network, self).__init__()
        #self.fc1 = nn.Linear(input_size, 128)
        #self.fc2 = nn.Linear(128, 64)
        #self.fc3 = nn.Linear(64, output_size)
        
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        print(self.device)
        input()

    def forward(self, x):
        #x = torch.relu(self.fc1(x))
        #x = torch.relu(self.fc2(x))
        #x = self.fc3(x)
        #return x
        return self.fc(x)

class target_Q_network(nn.Module):
    def __init__(self, input_size, output_size):
        super(target_Q_network, self).__init__()
        #self.fc1 = nn.Linear(input_size, 64)
        #self.fc2 = nn.Linear(64, 32)
        #self.fc3 = nn.Linear(32, output_size)
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        return self.fc(x)
        
        #x = torch.relu(self.fc1(x))
        #x = torch.relu(self.fc2(x))
        #x = self.fc3(x)
        #return x

class replay_buffer():
    def __init__(self, capacity, buffer):
        self.capacity = capacity
        self.buffer = buffer

    def add_experience(self, state, action, reward, next_state, done):
        
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def update_buffer_at(self, index, new_buffer):
        self.buffer[index] = new_buffer

    def sample_batch(self, batch_size):
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return states, actions, rewards, next_states, dones
    

actions = ["up","down","right","left","jump_up","jump_down","jump_right","jump_left"]
utils = common_functions()
path_to_file_maze = "./saved_maze/maze5"

deep_Q_learning_class(gamma=0.99, epsilon=0, batch_size = 32, load_maze=True, maze_args=[path_to_file_maze], actions=actions).vanillaDQN()
input()
batch_sizes = [32, 64]
gammas = [0.99]
epsilons = [0.15, 0.2]

Q_learning_scores = []
q_learning_comb = []
for batch_size in batch_sizes:
    for gamma in gammas:
        for epsilon in epsilons:
            q, policy = deep_Q_learning_class(gamma=gamma, epsilon=epsilon, batch_size = batch_size, load_maze=True, maze_args=[path_to_file_maze], actions=actions).DQN_agent()
            Q_learning_score = utils.play_episode(policy, path_to_file_maze)
            Q_learning_scores.append(Q_learning_score)
            q_learning_comb.append(("batch ",batch_size," gammas",gamma, " epsilon " ,epsilon," score ", Q_learning_score))

for v in q_learning_comb:

    print(v)
