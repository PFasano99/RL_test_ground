# ISPR project - Reinforcement Learning Play Ground

This repo is a playground to test some reinforcement learning algorithms on a labyrinth.
The reinforcement learning agents that are implemented to solve this task are the following:

- Actor-Critic
- Deep-Q Networks
- Value Iteration / Policy Iteration (Dynamic programming)
- SARSA / $\lambda$ SARSA / Q-Learning

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

## Getting started

The algorithms can be called from the main.py, they can be used directly on a pre-existing mazes or generating a new maze. 

The main.py has the following set of parameters:

<table>
    <thead>
        <tr>
            <th>Parameter</th>
            <th>Type</th>
            <th>Default</th>
            <th>Description</th>       
        </tr>
    </thead>
    <tbody>
        <tr><td class="tg-0lax" colspan="4">Parameters to choose which RL algorithms to run</td></tr>
        <tr>
            <td>--run_all</td>
            <td>bool</td>
            <td>True</td>
            <td>Run all RL algorithms on maze</td>
        </tr>
        <tr>
            <td>--run_AC</td>
            <td>bool</td>
            <td>False</td>
            <td>Run actor critic on maze</td>
        </tr>
        <tr>
            <td>--run_DQN</td>
            <td>bool</td>
            <td>False</td>
            <td>Run deep Q-network on maze</td>
        </tr>
        <tr>
            <td>--run_VI</td>
            <td>bool</td>
            <td>False</td>
            <td>Run value iteration on maze</td>
        </tr>
        <tr>
            <td>--run_SA</td>
            <td>bool</td>
            <td>False</td>
            <td>Run SARSA on maze</td>
        </tr>
        <tr>
            <td>--run_SAL</td>
            <td>bool</td>
            <td>False</td>
            <td>Run SARSA-lambda on maze</td>
        </tr>
        <tr>
            <td>--run_Q</td>
            <td>bool</td>
            <td>False</td>
            <td>Run Q-learning on maze</td>
        </tr>
        <tr>
            <td>--run_RA</td>
            <td>bool</td>
            <td>False</td>
            <td>Run random-approach on maze</td>
        </tr>
        <tr><td class="tg-0lax" colspan="4">Parameters for the maze</td></tr>
        <tr>
            <td>--maze_path</td>
            <td>str</td>
            <td>""</td>
            <td>Path to load an already existing maze.csv</td>
        </tr>
        <tr>
            <td>--maze_name</td>
            <td>str</td>
            <td>""</td>
            <td>The name to be assigned to the generated maze</td>
        </tr>
        <tr>
            <td>--save_path</td>
            <td>str</td>
            <td>""</td>
            <td>Path to save any file generated e.g. policies</td>
        </tr>
        <tr>
            <td>--maze_x</td>
            <td>int</td>
            <td>20</td>
            <td>Height of the maze</td>
        </tr>
        <tr>
            <td>--maze_y</td>
            <td>int</td>
            <td>40</td>
            <td>Width of the maze</td>
        </tr>
        <tr>
            <td>--maze_fp</td>
            <td>int</td>
            <td>1</td>
            <td>The number of minimum focal points to add in the maze</td>
        </tr>
        <tr><td class="tg-0lax" colspan="4">Parameters for deep models</td></tr>
        <tr>
            <td>--batch_size</td>
            <td>int</td>
            <td>512</td>
            <td>Batch size for deep models</td>
        </tr>
        <tr>
            <td>--epsilon_decay</td>
            <td>float</td>
            <td>0.999</td>
            <td>The value of decay of epsilon for each episode</td>
        </tr>
        <tr>
            <td>--episodes</td>
            <td>int</td>
            <td>2000</td>
            <td>Number of episodes to be run</td>
        </tr>
        <tr>
            <td>--replay_buffer_size</td>
            <td>int</td>
            <td>50000</td>
            <td>The size of the replay buffer from which the batches will be sampled</td>
        </tr>
        <tr>
            <td>--cuda</td>
            <td>bool</td>
            <td>True</td>
            <td>Use GPU for training</td>
        </tr>
        <tr>
            <td>--epsilon_start</td>
            <td>float</td>
            <td>1</td>
            <td>The epsilon starting value for the epsilon greedy choice of actions in the deep models</td>
        </tr>
        <tr><td class="tg-0lax" colspan="4">Parameters for RL</td></tr>
        <tr>
            <td>--gamma</td>
            <td>float</td>
            <td>0.99</td>
            <td>The discount value</td>
        </tr>
        <tr>
            <td>--epsilon</td>
            <td>float</td>
            <td>0.2</td>
            <td>The epsilon value for the epsilon greedy choice of actions (this parameter does not apply to deep models)</td>
        </tr>
        <tr>
            <td>--alpha</td>
            <td>float</td>
            <td>0.5</td>
            <td>Alpha therm for SARSA and Q-learning</td>
        </tr>
        <tr>
            <td>--lambda_s</td>
            <td>float</td>
            <td>0.2</td>
            <td>Lambda therm for SARSA-lambda</td>
        </tr>
    </tbody>
</table>

The use of run_all will use the same parameters for all the algorithms, this could lead to not finding the ideal solution.
    
    The parameters used to test the algorithm are the following:
    
    Random algorithm
        python3 ./main.py --maze_path ./saved_maze/maze4  --run_RA True --episodes 100
    
    Value iteration
        python3 ./main.py --maze_path ./saved_maze/maze4  --run_VI True --episodes 2000 --gamma 0.99
        
    SARSA 
        python3 ./main.py --maze_path ./saved_maze/maze4  --run_SA True --episodes 2000 --gamma 0.99 --alpha 0.4 --epsilon 0.2
    
    SARSA-lambda 
        python3 ./main.py --maze_path ./saved_maze/maze4  --run_SAL True --episodes 2000 --gamma 0.99 --alpha 0.6 --epsilon 0.2 --lambda_s 0.2
    
    Q-learning
        python3 ./main.py --maze_path ./saved_maze/maze4  --run_Q True --episodes 2000 --gamma 0.99 --alpha 0.5 --epsilon 0.2
    
    Deep Q-Network
        python3 ./main.py --maze_path ./saved_maze/maze4  --run_DQN True --episodes 2000 --gamma 0.99 --epsilon_start 1 --epsilon_decay 0.999 --cuda True --replay_buffer_size 50000 --batch_size 1024
    
    Actor Critic
        python3 ./main.py --maze_path ./saved_maze/maze4  --run_AC True --episodes 2000 --gamma 0.99 --epsilon_start 1 --epsilon_decay 0.999 --cuda True --replay_buffer_size 50000 --batch_size 1024
    