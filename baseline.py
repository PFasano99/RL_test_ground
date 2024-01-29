import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from maze_generator import *
from minosses_maze import *
from maze_ui import *
from utils import *

class baseline():

    def __init__(self, maze_args = [], actions= [], num_iterations = 100, player_char = 9):
        """
            actions is composed as [action_id,action_id,...] e.g. actions = ["up","down","right","left","jump_up","jump_down","jump_right","jump_left"]
        """
        
        self.actions = actions
        self.num_iteration = num_iterations
        self.player_char = player_char

        generator = maze_generator()
        
        print("loading maze")
        path = maze_args[0]
        maze, path, start_coord, finish_coord = generator.load_maze_from_csv(path)
        self.maze = maze
        self.path = path
        self.size_x = len(maze)
        self.size_y = len(maze[0])
        self.start_coord = start_coord
        self.finish_coord = finish_coord
    
    def solve_maze(self):
        utils = common_functions(self.maze)
        
        all_rewards = []
        all_steps = []
        all_wins = []

        for episode in tqdm(range(self.num_iteration)):
            state = self.start_coord
            total_reward = len(self.path)*0.4
            done = False
            steps = 0

            while True:
                action = np.random.choice(len(self.actions))
                next_state, reward = utils.action_value_function(state, self.actions[action], self.finish_coord)

                total_reward += reward
                steps += 1

                if next_state == self.finish_coord: 
                    done = True
                
                if done or steps > len(self.path)*10:
                    break

                state = next_state
            
            all_rewards.append(total_reward)
            all_steps.append(steps)
            all_wins.append(done)

            print(f"Episode: {episode}, Steps: {steps}, \nreward: {total_reward}, Win: {done}")

        print("Reward mean: ", np.mean(all_rewards))
        print("Steps mean: ", np.mean(all_steps))
        print("Wins: ", sum(1 for value in all_wins if value), "/",self.num_iteration)


path_to_file_maze = "./saved_maze/maze4"
actions = ["up","down","right","left"]#,"jump_up","jump_down","jump_right","jump_left"]

baseline(maze_args=[path_to_file_maze], actions=actions).solve_maze()

