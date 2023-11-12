# ISPR project - Reinforcement Learning Play Ground

This repo is a playground to test some reinforcement learning algorithms on a labyrinth.
The reinforcement learning agents that are implemented to solve this task are the following:

- Actor-Critic
- Deep-Q Networks
- Value Iteration (Dynamic programming)
- SARSA / $\lambda$ SARSA

## Requirements
Before running the code, install following requirements modules:

- numpy==1.21.0
- random2==1.0.1
- tqdm==4.61.1
- matplotlib==3.4.2
- torch==1.9.0
- keyboard==0.13.5
- pygame==2.0.1
    <br>
    All the requirements can be installed running:

        pip install -r requirements.txt

## The environment
The environment used in this task was custom made. It was created so that any labyrinth of size nxm (with n < m) could be automatically generated with different parameters to decide the complexity of the maze. The methods for generating the maze are located in maze_generator.py

    To generate a maze you can call the function as follows:
        
        from maze_generator import *

        generator = maze_generator()
        maze, path = maze.generate_maze(size_x= 20, size_y =40, start_coord = [0,1], finish_coord = [19,39], n_of_turns = 4, log = False)

    Or you can call the load function if you have a maze you have already generated and saved before.

        from maze_generator import *

        generator = maze_generator()
        maze, path, player_coords, win_cords = generator.load_maze_from_csv("saved_maze/maze1")

### Rewards and actions
The possible actions and rewards that the agent can take/gain are reported in the following table. In this environment the objective of the agent is to minimize the number of points he loses, the initial score is n_of_viable_states*0.4

<table>
        <thead>
            <tr>
                <th>Action ID</th>
                <th>Action</th>
                <th>Reward</th>
                <th>Reward on backtrack</th>
                <th>Reward on action fail</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>"up"</td>
                <td>Player treys to move up 1 step</td>
                <td>-0.05</td>
                <td>-0.25</td>
                <td>-0.75</td>
            </tr>
            <tr>
                <td>"down"</td>
                <td>Player treys to move down 1 step</td>
                <td>-0.05</td>
                <td>-0.25</td>
                <td>-0.75</td>
            </tr>
            <tr>
                <td>"left"</td>
                <td>Player treys to move left 1 step</td>
                <td>-0.05</td>
                <td>-0.25</td>
                <td>-0.75</td>
            </tr>
            <tr>
                <td>"right"</td>
                <td>Player treys to move right 1 step</td>
                <td>-0.05</td>
                <td>-0.25</td>
                <td>-0.75</td>
            </tr>
            <tr>
                <td>"jump_down"</td>
                <td>Player treys to jump down moving 2 positions down</td>
                <td>-0.15</td>
                <td>-0.25</td>
                <td>-0.75</td>
            </tr>
            <tr>
                <td>"jump_up"</td>
                <td>Player treys to jump up moving 2 positions up</td>
                <td>-0.15</td>
                <td>-0.25</td>
                <td>-0.75</td>
            </tr>
            <tr>
                <td>"jump_right"</td>
                <td>Player treys to jump right moving 2 positions right</td>
                <td>-0.15</td>
                <td>-0.25</td>
                <td>-0.75</td>
            </tr>
            <tr>
                <td>"jump_left"</td>
                <td>Player treys to jump left moving 2 positions left</td>
                <td>-0.15</td>
                <td>-0.25</td>
                <td>-0.75</td>
            </tr>
        </tbody>
    </table>

If the agent reaches the finishing state (win position) it will gain +1 point. 

An action is considered failed if the state after the action is the same; e.g 

    If the agent tries to go left but the only thing on his left is a wall this results in a failed action. 

        01000000             01000000
        09111111  --left-->  09111111
        00000001             00000001


The jump mechanic allows the agent to jump over walls, in other worlds it allows the agent to skip over 1 space; e.g.
        
        1000000                 1000000      1000111                 1000111
        1000100                 1000100      1000100                 1000100
        1911111 --jump_right--> 1119111  or  1190111 --jump_right--> 1110900 
        0000001                 0000001      1000100                 1000100
        0000001                 0000001      1111100                 1111100

***(where 1 is the legal path, 0  is a wall and 9 the player)***
