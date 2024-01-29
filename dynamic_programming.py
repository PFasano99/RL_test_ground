import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from maze_generator import *
from minosses_maze import *
from maze_ui import *
from utils import *

import random

class dynamic_programming():

    def __init__(self, discount_value, maze_path, actions= [], num_iterations = 100, player_char = 9, n_runs = 100):
        """
            actions is composed as [action_id,action_id,...] e.g. actions = ["up","down","right","left","jump_up","jump_down","jump_right","jump_left"]
        """
        
        self.gamma = discount_value
        self.actions = actions
        self.num_iteration = num_iterations
        self.player_char = player_char
        self.n_runs = n_runs

        generator = maze_generator()
        
        maze, path, start_coord, finish_coord = generator.load_maze_from_csv(maze_path)
        self.maze = maze
        self.path = path
        self.size_x = len(maze)
        self.size_y = len(maze[0])
        self.start_coord = start_coord
        self.finish_coord = finish_coord


class policy_iteration_class(dynamic_programming):
    def policy_iteration(self, theta=1e-6):
        print("Running policy iteration")
        utils = common_functions(self.maze)

        # Initialize a random policy
        policy = np.ones((len(self.path), len(self.actions))) / len(self.actions)

        iterate_policy = True
        while iterate_policy:
            # Policy Evaluation
            V = np.zeros([self.size_x, self.size_y])
            evaluating = True
            while evaluating:
                delta = 0
                for row in range(self.size_x):
                    for col in range(self.size_y):
                        state = [row,col]
                        if state in self.path:
                            v = V[state[0],state[1]]
                            bellman_expectation = 0
                            for action in self.actions:
                                next_state, reward = utils.action_value_function(state, action, self.finish_coord)
                                bellman_expectation += 1/len(self.actions) * (reward + self.gamma * V[next_state[0],next_state[1]])

                            V[state[0],state[1]] = bellman_expectation
                            delta = max(delta, abs(v - V[state[0],state[1]]))

                if delta < theta:
                    evaluating = False

            # Policy Improvement
            policy_stable = True
            for row in range(self.size_x):
                for col in range(self.size_y):
                    state = [row,col]
                    if state in self.path:
                        index_in_policy = utils.find_index_of_coordinate(self.path, state)
                        old_action = np.argmax(policy[index_in_policy])
                        action_values = np.zeros(len(self.actions))

                        a = 0
                        for action in self.actions:
                            next_state, reward = utils.action_value_function(state, action, self.finish_coord)
                            action_values[a] = 1/len(self.actions) * (reward + self.gamma * V[next_state[0],next_state[1]])
                            a += 1
                        best_action = np.argmax(action_values)

                        if old_action != best_action:
                            policy_stable = False

                        new_policy_row = np.zeros(len(self.actions))
                        new_policy_row[best_action] = 1
                        policy[index_in_policy] = new_policy_row

            if policy_stable:
                iterate_policy = False

        return policy, V

class value_iteration_class(dynamic_programming):
    def value_iteration(self, theta=1e-6):
        print("Running value iteration")
        utils = common_functions(self.maze)

        policy = np.zeros([len(self.path), len(self.actions)])
        state_values = np.zeros([self.size_x, self.size_y])
        delta = float("inf")
        
        done = False
        while delta > theta:
            delta = 0
            for row in range(self.size_x):
                for col in range(self.size_y):   
                    state = [row, col]
                    if state in self.path:

                        old_value = state_values[state[0],state[1]]
                        q_max = float("-inf")

                        a = 0
                        for action in self.actions:
                            next_state, reward = utils.action_value_function(state, action, self.finish_coord)
                            if reward > 0: done = True
                            value = reward + self.gamma * state_values[next_state[0],next_state[1]]
                            # Update the maximum Q-value and corresponding action probabilities
                            if value > q_max:
                                q_max = value
                                action_probs = np.zeros(len(self.actions))
                                action_probs[a] = 1
                            
                            a+=1
                        
                        # Update the state value with the maximum Q-value
                        state_values[state[0],state[1]] = q_max
                                                
                        state_id = utils.find_index_of_coordinate(self.path, state)
                        policy[state_id] = action_probs

                        # Update the delta with the maximum difference in state values
                        delta = max(delta, abs(old_value - state_values[state[0],state[1]]))
                        
        return policy, state_values

"""
actions = ["up","down","right","left"]#,"jump_up","jump_down","jump_right","jump_left"]

utils = common_functions()
path_to_file_maze = "./saved_maze/maze4"
val_it = value_iteration_class(discount_value=0.99, load_maze=True, maze_args=[path_to_file_maze], actions=actions)
policy, V = val_it.value_iteration()
val_iter_score = utils.play_episode(policy, path_to_file_maze, show_window = False)
#input()

pol_it = policy_iteration_class(discount_value=0.99, load_maze=True, maze_args=[path_to_file_maze], actions=actions)
policy, V = pol_it.policy_iteration()
pol_iter_score = utils.play_episode(policy, path_to_file_maze, show_window = False)

print("value iteration solved the maze with a score of: ", str(val_iter_score))
print("policy iteration solved the maze with a score of: ", str(pol_iter_score))
"""