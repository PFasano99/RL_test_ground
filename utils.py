import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import csv

from maze_generator import *
from minosses_maze import *
from maze_ui import *

class common_functions():
    def __init__(self, maze = [], player_char = 9):
        self.maze = maze
        if self.maze != []:
            self.size_x = len(maze)
            self.size_y = len(maze[0])
        else:
            self.size_x = 0
            self.size_y = 0
        self.player_char = player_char
        pass

    def action_value_function(self, state, action, finish_coord, final_state_reward = 0, pas_char = 1):
        """
            action represents the action id eg. "up", "down", etc
            This method returns the simulation of a specific action form a specific state
        """
        action_call = maze_actions(win_coords=finish_coord)
        maze_copy = np.copy(self.maze)
        return action_call.simulate_step(maze_copy, state, action=action)

    def find_index_of_coordinate(self, coords_list, target_coord):
        """
            Returns the index of the given coord from a list of coords.
        """
        for index, coord in enumerate(coords_list):
            if coord[0] == target_coord[0] and coord[1] == target_coord[1]:
                return index
        return -1

    def play_episode(self, policy, load_path = "", start_coord = [], finish_coord=[], maze_path=[], step_by_step = False, show_window = True):
        """
            Given the policy and either a path to a maze or the maze it self (with the start and finishing coords) this method 
            runs the policy over the maze showing the result of the policy and returning the score. 
        """
        if load_path != "":
            self.maze, maze_path, start_coord, finish_coord = maze_generator().load_maze_from_csv(load_path)
            self.size_x = len(self.maze)
            self.size_y = len(self.maze[0])

        if step_by_step: maze_generator().print_maze(self.maze, "int")

        player_coords = start_coord
        score = len(maze_path)*0.4

        if show_window: window = draw_matrix_grid(self.maze, size_x=self.size_x, size_y =self.size_y)
        actions = maze_actions(win_coords=finish_coord)

        pas_char = 2

        maze_copy = np.copy(self.maze)

        o = 1
        for state in self.maze:
            #print(state)
            maze_copy[state] = o
            o+=1

        if step_by_step: maze_generator().print_maze(maze_copy, "int")

        explored_path = []
        explored_path.append(player_coords)
        while True:   
            
            index_in_policy = self.find_index_of_coordinate(maze_path, player_coords)

            if(index_in_policy == -1):
                print("something went wrong")
                break
            
            action_to_take = policy[index_in_policy]
            
            if all(x == action_to_take[0] for x in action_to_take):
                action_index = random.randint(0, len(action_to_take) - 1)
            else:
                action_index = np.argmax(action_to_take)
            
            if step_by_step:
                print("---")
                print(policy[index_in_policy])
                print(index_in_policy)
                print(action_index)

            if action_index == 0:
                self.maze, pas_char, player_coords, cost = actions.go_up(self.maze, player_coords, pas_char, self.player_char, explored_path = explored_path)  
                score += cost           
                if show_window: window = update_ui(window, self.maze, size_x=self.size_x, size_y =self.size_y, score = score)
                
            elif action_index == 1:
                self.maze, pas_char, player_coords, cost = actions.go_down(self.maze, player_coords, pas_char, self.player_char, explored_path = explored_path)  
                score += cost
                if show_window: window = update_ui(window, self.maze, size_x=self.size_x, size_y =self.size_y, score = score)
                        
            elif action_index == 2:
                self.maze, pas_char, player_coords, cost = actions.go_right(self.maze, player_coords, pas_char, self.player_char, explored_path = explored_path)  
                score += cost
                if show_window: window = update_ui(window, self.maze, size_x=self.size_x, size_y =self.size_y, score = score)
                        
            elif action_index == 3:
                self.maze, pas_char, player_coords, cost = actions.go_left(self.maze, player_coords, pas_char, self.player_char, explored_path = explored_path)  
                score += cost      
                if show_window: window = update_ui(window, self.maze, size_x=self.size_x, size_y =self.size_y, score = score)
                            
            if action_index == 4:
                self.maze, pas_char, player_coords, cost = actions.jump(self.maze, player_coords, pas_char, self.player_char, 0, explored_path = explored_path, step_size = 2)  
                score += cost  
                if show_window: window = update_ui(window, self.maze, size_x=self.size_x, size_y =self.size_y, score = score)
                
            elif action_index == 5:
                self.maze, pas_char, player_coords, cost = actions.jump(self.maze, player_coords, pas_char, self.player_char, 2, explored_path = explored_path, step_size = 2)  
                score += cost
                if show_window: window = update_ui(window, self.maze, size_x=self.size_x, size_y =self.size_y, score = score)
                            
            elif action_index == 6:
                self.maze, pas_char, player_coords, cost = actions.jump(self.maze, player_coords, pas_char, self.player_char,1, explored_path = explored_path, step_size = 2)  
                score += cost
                if show_window: window = update_ui(window, self.maze, size_x=self.size_x, size_y =self.size_y, score = score)
                          
            elif action_index == 7:
                self.maze, pas_char, player_coords, cost = actions.jump(self.maze, player_coords, pas_char, self.player_char,3, explored_path = explored_path, step_size = 2)  
                score += cost           
                if show_window: window = update_ui(window, self.maze, size_x=self.size_x, size_y =self.size_y, score = score)

            explored_path.append(player_coords)
            
            if step_by_step:
                print("SCORE: ",score) 
                print("coords: ", player_coords)
                input()
            
            
            if score <= 0:
                if step_by_step: print("SORRY, YOU LOSE, SCORE: ", score)
                return score

            if step_by_step:
                print("--- --- --- --- --- --- --- --- --- ---")
                maze_generator().print_maze(self.maze, "int")
            
            time.sleep(0.2)#input()

            if player_coords == finish_coord:
                break
        if step_by_step: print("SCORE: ",score) 
        return score
    
    def save_dictionary_to_csv(self, data, filepath = "./"):
        # Flatten the dictionary for CSV
        flattened_data = [{'Key': key, 'Value': value} for key, value in data.items()]

        # Save the flattened dictionary to a CSV file
        with open(filepath, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['Key', 'Value'])

            # Write the header row
            writer.writeheader()

            # Write data rows
            writer.writerows(flattened_data)

    def load_dictionary_from_csv(self, loaded_data = {}, filepath = "./"):
        
        try:
            with open(filepath, 'r') as file:
                reader = csv.reader(file)
                next(reader)  # Skip the header row
                for row in reader:
                    # Split the key string into a tuple
                    key = eval(row[0])
                    value = float(row[1])
                    loaded_data[key] = value
        except FileNotFoundError:
            print(f"File {filepath} not found.")

        
        return loaded_data