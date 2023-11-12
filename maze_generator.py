import numpy as np
from random import randrange
import math
import csv


class maze_generator:

    def __init__(self):
        pass
    
    def generate_maze(self, size_x=70, size_y =100, start_coord = [0,1], finish_coord = [69,99], n_of_turns = 10, log = False):
        """
        
            Generate a maze of size_x * size_y the maze is matrix with 0 for wall and X for floor

            Params: 
                - size_x int for maze width (minimum 5)
                - size_y int for maze hight (minimum 5) 
                - start_coord the coordinates for the starting position (can't be the same as finish_coord)
                - finish_coord the coordinates for the finish position (can't be the same as start_coord)
                - n_of_turns int how many turns must there be between start and finish (max of turns is size_x)

        """ 

        if size_x < 5:
            size_x = 5
        if size_y < 5:
            size_y = 5

        if start_coord == finish_coord:
            if start_coord[0] == 0: #if the x coord is on the 0 or size_x than the start coords are either on the left or right wall
                finish_coord[0] = size_x
            elif start_coord[0] == size_x:
                finish_coord[0] = 0
            elif start_coord[1] == 0:
                finish_coord[1] = size_y
            elif start_coord[1] == size_y:
                finish_coord = 0 
            
            if size_x > n_of_turns:
                n_of_turns = size_x

        maze, path = self.base_maze(size_x=size_x, size_y=size_y, start_coord=start_coord, finish_coord=finish_coord, n_of_turns=n_of_turns)    
        maze, path = self.complicate_maze(size_x, size_y, maze, 15, path, mode = "closest")
    
        if log: 
            print("unpacking maze")
            print("---------------------")
            self.print_maze(maze)
        
        return maze, path

    def base_maze(self, size_x, size_y, start_coord = [0,1], finish_coord = [0,3], n_of_turns = 2, log = False):
        """
            Creates the direct path from start to finish
        """
        maze = np.zeros((size_x,size_y))
        maze[start_coord[0], start_coord[1]] = 3
        maze[finish_coord[0], finish_coord[1]] = 4
        
        turns_coords = []
        excluded_x = []
        i = 0
        while i < n_of_turns:
            x = randrange(1,size_x-1)
            y = randrange(1,size_y-1)
            if x not in excluded_x:
                excluded_x.append(x)
                new_coord = [x,y]
                turns_coords.append(new_coord)
                x,y = new_coord
                maze[x,y] = 2
                i+=1
        
        turns_coords = self.sort_coords_by_closeness(turns_coords, start_coord)
        
        if log: 
            print("start: ",start_coord, " end: ",finish_coord)
            print("turns_coords in order: ", turns_coords)

        path = [[start_coord[0],start_coord[1]]]
        if i > 0:
            for i in range (0,n_of_turns):
                if i == 0:
                    maze, path_added = self.draw_path_between_coords([[start_coord[0],start_coord[1]], [turns_coords[i][0],turns_coords[i][1]]], maze)
                    path.extend(path_added)
                
                    if log: 
                        print("connecting s: ", start_coord, " to ", turns_coords[i])  
                        print("path in a ", path)

                else:
                    maze, path_added = self.draw_path_between_coords([[turns_coords[i-1][0],turns_coords[i-1][1]], [turns_coords[i][0],turns_coords[i][1]]], maze)
                    path.extend(path_added) 
                    
                    if log: 
                        print("connecting: ", turns_coords[i-1], " to ", turns_coords[i])            
                        print("path in b ", path)

            maze, path_added = self.draw_path_between_coords([[turns_coords[-1][0],turns_coords[-1][1]], [finish_coord[0],finish_coord[1]]], maze)
            path.extend(path_added) 
            
            if log:
                print("connecting: ", turns_coords[-1], " to ", finish_coord)
                print("path in c ", path)
            

        else:
            
            maze, path_added = self.draw_path_between_coords([[start_coord[0],start_coord[1]], [finish_coord[0],finish_coord[1]]], maze)
            path.extend(path_added) 
            
            if log:        
                print("connecting: ", start_coord, " to ", finish_coord)
                print("path in d ", path)


        maze[finish_coord[0],finish_coord[1]] = 4

        if len(turns_coords) > 0:
            for turn in turns_coords:
                maze[turn[0],turn[1]] = 6
                

        return maze, path

    def complicate_maze(self, size_x, size_y, maze, n_new_points, path, mode = "random", log = False):
        """
            adds new paths in the maze adding to the complexity of navigating it.
        """
        original_path = path.copy()
        i = 0
        while i < n_new_points:
            x = randrange(1,size_x-1)
            y = randrange(1,size_y-1)
            start_point = [x,y]
            
            if start_point not in path:      
                if mode == "random":
                    finish_point = path[randrange(0,len(path)-1)]   
                    maze[finish_point[0],finish_point[1]] = 5        
                elif mode == "closest":
                    ordered_path = self.sort_coords_by_closeness(original_path, start_point)
                    finish_point = ordered_path[0]

                if log:
                    print("start ",start_point)
                    print("finish ",finish_point)

                maze, path_added = self.draw_path_between_coords([[start_point[0],start_point[1]], [finish_point[0],finish_point[1]]], maze=maze)
                path.extend(path_added) 
                maze[x,y] = 5
                
        
                i+=1
                
        return maze, path 

    def draw_path_between_coords(self, points = [], maze=[], path_char=1, log = False):
        path = []
        ordered_points = self.sort_points(points)
        
        if log: print("sorted-points: ", ordered_points)
        
        if ordered_points[0][0] == ordered_points[1][0]:
            dir = 1
            if ordered_points[0][1] > ordered_points[1][1]:
                dir = -1
                if log: print("DIRECT CONNECTION A-1")
            else: 
                if log: print("DIRECT CONNECTION A")
                
            maze, path_to_add = self.draw_horizontal_line(ordered_points[0], ordered_points[1], maze, path_char = path_char, direction=dir)
            path.extend(path_to_add)

            return maze, path
            
        elif ordered_points[0][1] == ordered_points[1][1]:
            dir = 1
            if ordered_points[0][0] > ordered_points[1][0]:
                dir = -1
                if log: print("DIRECT CONNECTION B-1")
            else: 
                if log: print("DIRECT CONNECTION B-1")
            maze, path_to_add = self.draw_vertical_line(ordered_points[0], ordered_points[1], maze, path_char = path_char, direction=dir)
            path.extend(path_to_add)  
    
            return maze, path
       
        mid_point = []    
        direction = randrange(0,2)


        if ordered_points[0][0] < ordered_points[1][0] and ordered_points[0][1] > ordered_points[1][1]:
            direction = 0
        elif ordered_points[0][0] < ordered_points[1][0] and ordered_points[0][1] < ordered_points[1][1]:
            direction = 1
        

        if direction == 0:
            mid_point = [ordered_points[0][0], ordered_points[1][1]]
            if log: print("connection with midpoint: ", mid_point, " type A-",direction)
            
            new_coords = ordered_points[0]
            dir = 1
            if mid_point[0] == ordered_points[0][0] and mid_point[1] < ordered_points[0][1]:
                dir = -1
            if log: print("connecting inside ",new_coords," to ", mid_point)
            maze, path_to_add = self.draw_horizontal_line(new_coords, mid_point, maze, direction=dir, path_char = path_char)
            path.extend(path_to_add)
            
            new_coords = [mid_point[0], mid_point[1]]
            maze, path_to_add = self.draw_vertical_line(new_coords, ordered_points[1], maze, path_char = path_char)
            path.extend(path_to_add)      

        else:
            mid_point = [ordered_points[1][0], ordered_points[0][1]]
            if log: print("connection with midpoint: ", mid_point, " type B-",direction)
            
            new_coords = ordered_points[0]
            if log: print("connect inner B: ", new_coords, " to ", mid_point)
            maze, path_to_add = self.draw_vertical_line(new_coords, mid_point, maze, path_char = path_char)
            path.extend(path_to_add)
            
            new_coords = [mid_point[0], mid_point[1]]
            maze, path_to_add = self.draw_horizontal_line(new_coords, ordered_points[1], maze, path_char = path_char)
            path.extend(path_to_add)

        maze[mid_point[0],mid_point[1]] = 2
        
        return maze, path

    def draw_horizontal_line(self, point1, point2, maze, path_char=1, direction=1):
        #direction 1 means from going left to right, direction -1 right to left
        path = []
        new_coords = point1
        while new_coords != point2:
            new_coords[1] += direction
            #print("a " ,new_coords)
            if new_coords[1] >= len(maze[1])-1 or new_coords[1] < 0:
                break
            path.append([new_coords[0], new_coords[1]])
            maze[new_coords[0],new_coords[1]] = path_char
        
        return maze, path

    def draw_vertical_line(self, point1, point2, maze, path_char=1, direction=1):
    # direction 1 means from top to bottom, direction -1 is bottom to top
        path = []
        new_coords = point1
        while new_coords != point2:
            new_coords[0] += direction
            if new_coords[0] >= len(maze[0])-1 or new_coords[0] < 0:
                break
            path.append([new_coords[0], new_coords[1]])
            maze[new_coords[0],new_coords[1]] = path_char
        
        return maze, path

    def print_maze(self, maze, type_is = "int"):
        print("printing maze")
        # Iterate through rows and columns and print each element
        i = 0
        for row in maze:
            print(i, end=" ")
            i+=1
            for element in row:
                if type_is == "int":
                    print(int(element), end=" ")  # Use end=" " to print elements in the same row with space
                elif type_is == "float":
                    print(float(element), end=" ")  # Use end=" " to print elements in the same row with space
                    
            print()  # Print a newline to move to the next row

    def sort_points(self, points, ordering = "descend", axis = 0):

        if ordering == "ascend":
            if axis == 0:
                sorted_points = sorted(points, key=lambda point: point[0], reverse=True)
            elif axis == 1:
                sorted_points = sorted(points, key=lambda point: point[1], reverse=True)
        elif ordering == "descend":     
            if axis == 0:
                sorted_points = sorted(points, key=lambda point: point[0], reverse=False)
            elif axis == 1:
                sorted_points = sorted(points, key=lambda point: point[1], reverse=False)

        return sorted_points

    def euclidean_distance(self, coord_A = [0,0], coord_B = [1,1]):
        return math.sqrt((coord_A[0] - coord_B[0])**2 + (coord_A[1] - coord_B[1])**2)

    def sort_coords_by_closeness(self, coords, starting_coord):
        coords.sort(key=lambda coord: self.euclidean_distance(coord, starting_coord))
        return coords

    def save_maze_as_csv(self, maze, filename, save_as = "int"):
        try:
            with open(filename, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                for row in maze:
                    if save_as == "int":
                        csvwriter.writerow(map(int, row))
                    elif save_as == "float":
                        csvwriter.writerow(map(float, row))
            print(f"maze saved to {filename} as CSV")
        except Exception as e:
            print(f"Error while saving the maze as CSV: {e}")

    def load_maze_from_csv(self, filename, load_as = "int"):
        try:
            maze = []
            with open(filename, 'r') as csvfile:
                csvreader = csv.reader(csvfile)
                for row in csvreader:
                    if load_as == "int":
                        row = list(map(int, row))
                    elif load_as == "float":
                        row = list(map(float, row))
                    maze.append(row)
            maze = np.array(maze)
            path, start_coord, finish_coord = self.find_non_zero_coordinates(maze)
            return maze, path, start_coord, finish_coord
        except FileNotFoundError:
            print(f"File {filename} not found.")
        except Exception as e:
            print(f"Error while loading the maze from CSV: {e}")
        return None, None, None, None

    def find_non_zero_coordinates(self, maze, start_char = 3, finish_char = 4):
        coordinates = []
        start_coord = []
        finish_coord = []
        for i, row in enumerate(maze):
            for j, value in enumerate(row):
                if value != 0:
                    coordinates.append([i, j])
                    if value == start_char:
                        start_coord = [i,j]
                    elif value == finish_char:
                        finish_coord = [i,j]
        return coordinates, start_coord, finish_coord
