import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from maze_generator import *
from minosses_maze import *
from maze_ui import *
from utils import *

import random

class dynamic_programming():

    def __init__(self, discount_value, load_maze = False, maze_args = [], actions= [], num_iterations = 100, player_char = 9, n_runs = 100):
        """
            actions is composed as [action_id,action_id,...] e.g. actions = ["up","down","right","left","jump_up","jump_down","jump_right","jump_left"]
        """
        
        self.gamma = discount_value
        self.actions = actions
        self.num_iteration = num_iterations
        self.player_char = player_char
        self.n_runs = n_runs


        generator = maze_generator()
        
        if load_maze:
            print("loading maze")
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

class policy_iteration_class(dynamic_programming):
    def policy_iteration(self, theta=1e-6):
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
                            index_in_policy = utils.find_index_of_coordinate(self.path, state)
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
                                a+=1
                            best_action = np.argmax(action_values)

                            if old_action != best_action:
                                policy_stable = False

                            policy[index_in_policy] = np.zeros(len(self.actions)) 
                            policy[index_in_policy, best_action] = 1
                    
        
                        if policy_stable:
                            iterate_policy = False

        maze_generator().save_maze_as_csv(V, "./saved_maze/maze_pol_iter_"+str(0), "float")
        maze_generator().save_maze_as_csv(policy, "./saved_maze/policy_pol_iter_"+str(0), "float")
        return policy, V

class value_iteration_class(dynamic_programming):
    def value_iteration(self, theta=1e-6):
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
                            if value > q_max:
                                q_max = value
                                action_probs = np.zeros(len(self.actions))
                                action_probs[a] = 1
                            
                            a+=1
                            
                        state_values[state[0],state[1]] = q_max
                                                
                        state_id = utils.find_index_of_coordinate(self.path, state)
                        policy[state_id] = action_probs

                        delta = max(delta, abs(old_value - state_values[state[0],state[1]]))

        
        print("--- Iteration ---")
        maze_generator().print_maze(state_values, "float")
        print("---- Policy ----")
        maze_generator().print_maze(policy, "float")
  
        maze_generator().save_maze_as_csv(state_values, "./saved_maze/maze_val_iter_"+str(0), "float")
        maze_generator().save_maze_as_csv(policy, "./saved_maze/policy_val_iter_"+str(0), "float")
        return policy, state_values

actions = ["up","down","right","left","jump_up","jump_down","jump_right","jump_left"]

utils = common_functions()
path_to_file_maze = "./saved_maze/maze3"
val_it = value_iteration_class(discount_value=0.99, load_maze=True, maze_args=[path_to_file_maze], actions=actions)
policy, V = val_it.value_iteration()
val_iter_score = utils.play_episode(policy, path_to_file_maze)
input()

pol_it = policy_iteration_class(discount_value=0.99, load_maze=True, maze_args=[path_to_file_maze], actions=actions)
policy, V = pol_it.policy_iteration()
pol_iter_score = utils.play_episode(policy, path_to_file_maze)

print("value iteration solved the maze with a score of: ", str(val_iter_score))
print("policy iteration solved the maze with a score of: ", str(pol_iter_score))