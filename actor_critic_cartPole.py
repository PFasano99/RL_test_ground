import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import random

# Define the actor-critic network
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc_pi = nn.Linear(64, action_dim)
        self.fc_v = nn.Linear(64, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        pi = torch.softmax(self.fc_pi(x), dim=0)
        v = self.fc_v(x)
        return pi, v

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

# Define the environment and other parameters
env = gym.make('CartPole-v1')
num_episodes = 1500
discount_factor = 0.99
learning_rate = 0.001
epsilon_start = 1.0
epsilon_decay = 0.997
epsilon_min = 0.01
replay_capacity = 10000  # Set your desired replay buffer capacity

# Initialize the ActorCritic network and the experience buffer
agent = ActorCritic(env.observation_space.shape[0], env.action_space.n)
replay_buffer = replay_buffer(capacity=replay_capacity, buffer=[])

# Define the optimizer
optimizer = optim.Adam(agent.parameters(), lr=learning_rate)

all_rewards = []

# Define the training loop
for episode in tqdm.tqdm(range(num_episodes), desc="Running a2c episodes"):
    # Initialize the environment
    state, _ = env.reset()
    done = False
    total_reward = 0

    epsilon = max(epsilon_start * epsilon_decay**episode, epsilon_min)

    while not done:
        # Select an action using epsilon-greedy strategy
        if np.random.rand() < epsilon:
            action = np.random.choice(env.action_space.n)
            probs = np.zeros(env.action_space.n)
            probs[action] = 1/env.action_space.n
            probs = torch.tensor(probs, dtype=torch.float32)
            
        else:
            probs, _ = agent(torch.tensor(state, dtype=torch.float32))
            action = np.argmax(probs.detach().numpy())

        # Take a step in the environment
        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward

        # Add the experience to the replay buffer
        if done == True:
            done = 1
        else:
            done = 0
        replay_buffer.add_experience(state, action, reward, next_state, done)

        if (len(replay_buffer) > 32):
            # Sample a batch from the replay buffer
            batch_size = min(len(replay_buffer), 32)  # Adjust batch size as needed
            states, actions, rewards, next_states, dones = replay_buffer.sample_batch(batch_size)
            states = np.asarray(states)
            next_states = np.asarray(next_states)
            # Calculate the TD error and loss
            _, next_val = agent(torch.tensor(next_states, dtype=torch.float32))
            vals = agent(torch.tensor(states, dtype=torch.float32))[1]

            errs = []
            for i in range(len(rewards)):
                errs = rewards[i] + discount_factor * (next_val * (1 - dones[i])) - vals
            
            actor_losses = []
            probs, _ = agent(torch.tensor(states, dtype=torch.float32))

            for probs in probs:
                for action, err in zip(actions, errs):
                    loss_term = -torch.log(probs[action]) * err
                    actor_losses.append(loss_term)
            critic_losses = [torch.square(err) for err in errs]
            loss = sum(actor_losses) + sum(critic_losses)
        else:
             # Take a step in the environment
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward

            # Calculate the TD error and loss
            _, next_val = agent(torch.tensor(next_state, dtype=torch.float32))
            err = reward + discount_factor * (next_val * (1 - done)) - agent(torch.tensor(state, dtype=torch.float32))[1]
            actor_loss = -torch.log(probs[action]) * err
            critic_loss = torch.square(err)
            loss = actor_loss + critic_loss
        if  done: break
        # Update the network
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Set the state to the next state
        state = next_state

    if episode % 50 == 0:
        print("epsilon = ", epsilon)
    # Print the total reward for the episode
    print(f'Episode {episode}: Total reward = {total_reward}')
    all_rewards.append(total_reward)
print("fnal epsilon = ", epsilon)

print("max reward is: ", np.max(all_rewards), " at ", (np.argmax(all_rewards) + 1))
