from maze_generator import *
from maze_ui import *
import time
import keyboard

def play(size_x = 20, size_y=40, player_char = 9, pas_char = 3, player_starting_coords= [0,1], win_cords = [19,39], save = False, load = True):
    """
        This function is used to play the Minosse's maze
    """

    player_coords = player_starting_coords
    key_pressed = False

    generator = maze_generator()

    if load:
        maze, path, player_coords, win_cords = generator.load_maze_from_csv("saved_maze/maze4")
        size_x = len(maze)
        size_y = len(maze[0])
    else:
        maze, path = generator.generate_maze(size_x=size_x, size_y =size_y, start_coord = player_coords, finish_coord = win_cords, n_of_turns = 4, log = False)

    if save: generator.save_maze_as_csv(maze, "saved_maze/maze6")

    score = len(path)*0.4
    window = draw_matrix_grid(maze, size_x=size_x, size_y =size_y)

    space_bar_pressed = False

    actions = maze_actions(win_coords=win_cords)
    explored_path = []
    explored_path.append(player_coords)
    while True:   

        if keyboard.is_pressed("spacebar") and key_pressed == False:
            space_bar_pressed = not space_bar_pressed
            key_pressed = True

        if space_bar_pressed:
            if keyboard.is_pressed("w"):
                print("You pressed jump 'w'.")
                maze, pas_char, player_coords, cost = actions.jump(maze, player_coords, pas_char, player_char, 0, explored_path = explored_path, step_size = 2)  
                score += cost
                window = update_ui(window, maze, size_x=size_x, size_y =size_y, score = score)
                key_pressed = True
                space_bar_pressed = False
                
            elif keyboard.is_pressed("s"):
                print("You pressed jump 's'.")
                maze, pas_char, player_coords, cost = actions.jump(maze, player_coords, pas_char, player_char, 2, explored_path = explored_path, step_size = 2)  
                score += cost
                window = update_ui(window, maze, size_x=size_x, size_y =size_y, score = score)
                key_pressed = True
                space_bar_pressed = False
                
            elif keyboard.is_pressed("d"):
                print("You pressed jump 'd'.")
                maze, pas_char, player_coords, cost = actions.jump(maze, player_coords, pas_char, player_char, 1, explored_path = explored_path, step_size = 2)  
                score += cost
                window = update_ui(window, maze, size_x=size_x, size_y =size_y, score = score)
                key_pressed = True
                space_bar_pressed = False
                        
            elif keyboard.is_pressed("a"):
                print("You pressed jump 's'.")
                maze, pas_char, player_coords, cost = actions.jump(maze, player_coords, pas_char, player_char, 3, explored_path = explored_path, step_size = 2)  
                score += cost
                window = update_ui(window, maze, size_x=size_x, size_y =size_y, score = score)
                key_pressed = True   
                space_bar_pressed = False

        if keyboard.is_pressed("w") and space_bar_pressed == False and key_pressed == False:
            print("You pressed 'w'.")
            maze, pas_char, player_coords, cost = actions.go_up(maze, player_coords, pas_char, player_char, explored_path = explored_path)  
            score += cost
            window = update_ui(window, maze, size_x=size_x, size_y =size_y, score = score)
            key_pressed = True      
            
        elif keyboard.is_pressed("s") and space_bar_pressed == False and key_pressed == False:
            print("You pressed 's'.")
            maze, pas_char, player_coords, cost = actions.go_down(maze, player_coords, pas_char, player_char, explored_path = explored_path)  
            score += cost
            window = update_ui(window, maze, size_x=size_x, size_y =size_y, score = score)
            key_pressed = True
            
                     
        elif keyboard.is_pressed("d") and space_bar_pressed == False and key_pressed == False:
            print("You pressed 'd'.")
            maze, pas_char, player_coords, cost = actions.go_right(maze, player_coords, pas_char, player_char, explored_path = explored_path)  
            score += cost
            window = update_ui(window, maze, size_x=size_x, size_y =size_y, score = score)
            key_pressed = True
                     
        elif keyboard.is_pressed("a") and space_bar_pressed == False and key_pressed == False:
            print("You pressed 's'.")
            maze, pas_char, player_coords, cost = actions.go_left(maze, player_coords, pas_char, player_char, explored_path = explored_path)  
            score += cost
            window = update_ui(window, maze, size_x=size_x, size_y =size_y, score = score)
            key_pressed = True      
             
        elif keyboard.is_pressed("esc"):
            print("BYE")
            break
        
        if key_pressed:
            explored_path.append(player_coords)
            if actions.check_win(player_coords, win_cords):
                score+=10
                window = update_ui(window, maze, size_x=size_x, size_y =size_y, score = score)
                break
            print("SCORE: ", score)
            time.sleep(0.2)
            key_pressed = False
    
    print("SCORE: ",score)     

class maze_actions():
        
    def __init__(self, win_coords = [], cost_for_step = -0.05, cost_for_backtrack = -0.25, cost_for_jump = -0.15, cost_for_illegal_move = -0.75, reward_for_winning = 1):
        self.cost_for_step = cost_for_step    
        self.cost_for_backtrack = cost_for_backtrack     
        self.cost_for_jump = cost_for_jump    
        self.cost_for_illegal_move = cost_for_illegal_move     
        self.reward_for_winning = reward_for_winning  
        self.win_coords = win_coords

    def simulate_step(self, maze, state, action, explored_path = [], step_size = 1, override_cost = [False, 0]):    
        """
            Given a maze, a state and an action this method simulates and returns what would happen (next state and reward) if a specific action is taken
        """
        if action == "up":
            _, _, next_state, reward = self.go_up(maze, state, 1, 9, explored_path = explored_path, step_size = step_size, override_cost = override_cost)
        elif action == "down":
            _, _, next_state, reward = self.go_down(maze, state, 1, 9, explored_path = explored_path, step_size = step_size, override_cost = override_cost)
        elif action == "left":
            _, _, next_state, reward = self.go_left(maze, state, 1, 9, explored_path = explored_path, step_size = step_size, override_cost = override_cost)
        elif action == "right":
            _, _, next_state, reward = self.go_right(maze, state, 1, 9, explored_path = explored_path, step_size = step_size, override_cost = override_cost)
        elif action == "jump_up":
            _, _, next_state, reward = self.jump(maze, state, 1, 9, 0, explored_path = explored_path, step_size = 2, override_cost = override_cost)
        elif action == "jump_down":
            _, _, next_state, reward = self.jump(maze, state, 1, 9, 2, explored_path = explored_path, step_size = 2, override_cost = override_cost)
        elif action == "jump_left":
            _, _, next_state, reward = self.jump(maze, state, 1, 9, 3, explored_path = explored_path, step_size = 2, override_cost = override_cost)
        elif action == "jump_right":
            _, _, next_state, reward = self.jump(maze, state, 1, 9, 1, explored_path = explored_path, step_size = 2, override_cost = override_cost)
        
        return next_state, reward

    def go_up(self, maze, current_player_coords, previous_char, player_char, explored_path = [], step_size = 1, override_cost = [False, 0]):
        
        if current_player_coords[0] - step_size >= 0 and current_player_coords[0] + step_size < len(maze): 
            coords_to_check = [current_player_coords[0]-step_size,current_player_coords[1]]
            if maze[coords_to_check[0],coords_to_check[1]] != 0:
                current_char = maze[coords_to_check[0],coords_to_check[1]]
                maze[coords_to_check[0],coords_to_check[1]] = player_char
                maze[current_player_coords[0],current_player_coords[1]] = previous_char
                
                if coords_to_check[0] == self.win_coords[0] and coords_to_check[1] == self.win_coords[1]:
                    return maze, current_char, [coords_to_check[0],coords_to_check[1]], self.reward_for_winning

                if [coords_to_check[0],coords_to_check[1]] in explored_path:
                    return maze, current_char, [coords_to_check[0],coords_to_check[1]], self.cost_for_backtrack
                
                if override_cost[0]:
                    return maze, current_char, [coords_to_check[0],coords_to_check[1]], override_cost[1]
                    
                return maze, current_char, [coords_to_check[0],coords_to_check[1]], self.cost_for_step
            
        if override_cost[0]:
            return maze, previous_char, current_player_coords, override_cost[1]*2

        return maze, previous_char, current_player_coords, self.cost_for_illegal_move

    def go_down(self, maze, current_player_coords, previous_char, player_char, explored_path = [], step_size = 1, override_cost = [False,0]):
        
        if current_player_coords[0] + step_size >= 0 and current_player_coords[0] + step_size < len(maze): 
            coords_to_check = [current_player_coords[0]+step_size,current_player_coords[1]]
            if maze[coords_to_check[0],coords_to_check[1]] != 0:
                current_char = maze[coords_to_check[0],coords_to_check[1]]
                maze[coords_to_check[0],coords_to_check[1]] = player_char
                maze[current_player_coords[0],current_player_coords[1]] = previous_char

                if coords_to_check[0] == self.win_coords[0] and coords_to_check[1] == self.win_coords[1]:
                    return maze, current_char, [coords_to_check[0],coords_to_check[1]], self.reward_for_winning
                
                if [coords_to_check[0],coords_to_check[1]] in explored_path:
                    return maze, current_char, [coords_to_check[0],coords_to_check[1]], self.cost_for_backtrack
                
                if override_cost[0]:
                    return maze, current_char, [coords_to_check[0],coords_to_check[1]], override_cost[1]

                return maze, current_char, [coords_to_check[0],coords_to_check[1]], self.cost_for_illegal_move

        if override_cost[0]:
            return maze, previous_char, current_player_coords, override_cost[1]*2
        
        return maze, previous_char, current_player_coords, self.cost_for_step*2
        
    def go_right(self, maze, current_player_coords, previous_char, player_char, explored_path = [], step_size = 1, override_cost=[False,0]):
        
        if current_player_coords[1] + step_size >= 0 and current_player_coords[1] + step_size < len(maze[1]): 
            coords_to_check = [current_player_coords[0],current_player_coords[1]+step_size]
            if maze[coords_to_check[0],coords_to_check[1]] != 0:
                current_char = maze[coords_to_check[0],coords_to_check[1]]
                maze[coords_to_check[0],coords_to_check[1]] = player_char
                maze[current_player_coords[0],current_player_coords[1]] = previous_char

                if coords_to_check[0] == self.win_coords[0] and coords_to_check[1] == self.win_coords[1]:
                    return maze, current_char, [coords_to_check[0],coords_to_check[1]], self.reward_for_winning

                if [coords_to_check[0],coords_to_check[1]] in explored_path:
                    return maze, current_char, [coords_to_check[0],coords_to_check[1]], self.cost_for_backtrack

                if override_cost[0]:
                    return maze, current_char, [coords_to_check[0],coords_to_check[1]], override_cost[1]
        
                return maze, current_char, [coords_to_check[0],coords_to_check[1]], self.cost_for_step
        
        if override_cost[0]:
            return maze, previous_char, current_player_coords, override_cost[1]*2
        
        return maze, previous_char, current_player_coords, self.cost_for_illegal_move

    def go_left(self, maze, current_player_coords, previous_char, player_char, explored_path = [], step_size = 1, override_cost = [False,0]):
        
        if current_player_coords[1] - step_size >= 0 and current_player_coords[1] - step_size < len(maze[1]): 
            coords_to_check = [current_player_coords[0],current_player_coords[1]-step_size]
            if maze[coords_to_check[0],coords_to_check[1]] != 0:
                current_char = maze[coords_to_check[0],coords_to_check[1]]
                maze[coords_to_check[0],coords_to_check[1]] = player_char
                maze[current_player_coords[0],current_player_coords[1]] = previous_char

                if coords_to_check[0] == self.win_coords[0] and coords_to_check[1] == self.win_coords[1]:
                    return maze, current_char, [coords_to_check[0],coords_to_check[1]], self.reward_for_winning

                if [coords_to_check[0],coords_to_check[1]] in explored_path:
                    return maze, current_char, [coords_to_check[0],coords_to_check[1]], self.cost_for_backtrack

                if override_cost[0]:
                    return maze, current_char, [coords_to_check[0],coords_to_check[1]], override_cost[1]

                return maze, current_char, [coords_to_check[0],coords_to_check[1]], self.cost_for_step
        
        if override_cost[0]:
            return maze, previous_char, current_player_coords, override_cost[1]*2

        return maze, previous_char, current_player_coords, self.cost_for_illegal_move

    def jump(self, maze, current_player_coords, previous_char, player_char, direction, explored_path = [], step_size = 2, override_cost = [False,0]):
        """
            where direction 
                - 0 is up
                - 1 is left
                - 2 is down
                - 3 is right
        """

        if direction == 0:
            maze, previous_char, current_player_coords, cost = self.go_up(maze, current_player_coords, previous_char, player_char, explored_path = explored_path, step_size = 2, override_cost = [True, self.cost_for_jump])
        elif direction == 1:
            maze, previous_char, current_player_coords, cost = self.go_right(maze, current_player_coords, previous_char, player_char, explored_path = explored_path, step_size = 2, override_cost = [True, self.cost_for_jump]) 
        elif direction == 2:
            maze, previous_char, current_player_coords, cost = self.go_down(maze, current_player_coords, previous_char, player_char, explored_path = explored_path, step_size = 2, override_cost = [True, self.cost_for_jump])
        elif direction == 3:
            maze, previous_char, current_player_coords, cost = self.go_left(maze, current_player_coords, previous_char, player_char, explored_path = explored_path, step_size = 2, override_cost = [True, self.cost_for_jump])

        return maze, previous_char, current_player_coords, cost

    def check_win(self, current_player_coords, win_coords):
        if current_player_coords == win_coords:
            print("You reached the end")
            return True
        return False
    

#play()
