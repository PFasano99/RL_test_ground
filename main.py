import actor_critic
import baseline
import DeepQNetwork
import dynamic_programming
import temporal_difference
import utils
import maze_generator

import argparse


def main():

    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description='This script is made as an interface to call the RL algorithms: \n - DQN, \n - AC, \n - value iteration, \n - SARSA, \n - SARSA-lambda. ')

    # general params
    parser.add_argument('--save_path', type=str, default="", help='Path to save any file generated e.g. policies')

    # parameters to choose which RL algorithms to run
    parser.add_argument('--run_all', type=bool, default=True, help='Run all RL algorithms on maze')
    parser.add_argument('--run_AC', type=bool, help='Run actor critic on maze')
    parser.add_argument('--run_DQN', type=bool, help='Run deep Q-network on maze')
    parser.add_argument('--run_VI', type=bool, help='Run value iteration on maze')
    parser.add_argument('--run_SA', type=bool, help='Run SARSA on maze')
    parser.add_argument('--run_SAL', type=bool, help='Run SARSA-lambda on maze')
    parser.add_argument('--run_Q', type=bool, help='Run Q-learning on maze')
    parser.add_argument('--run_RA', type=bool, help='Run random-approach on maze')
    # parameters for maze maze
    parser.add_argument('--maze_path', type=str, default="", help='Path to load an already existing maze.csv')
    parser.add_argument('--maze_name', type=str, default="", help='The name to be assigned to the generated maze')
    parser.add_argument('--maze_x', type=int, help='hight of the maze')
    parser.add_argument('--maze_y', type=int, help='width of the maze')
    parser.add_argument('--maze_fp', type=int, default = 1, help='the number of minimum focal points to add in the maze')
    # parameters for deep models
    parser.add_argument('--batch_size', type=int, default=512, help='batch size for deep models')
    parser.add_argument('--epsilon_decay', type=float, default=0.999, help='the value of decay of epsilon for each episode')
    parser.add_argument('--episodes', type=int, default=2000, help='number of episodes to be run')
    parser.add_argument('--replay_buffer_size', type=int, default=50000, help='the size of the replay buffer from which the batches will be sampled')
    parser.add_argument('--cuda', type=bool, default=True, help='use GPU for training')
    parser.add_argument('--epsilon_start', type=float, default=1, help='the epsilon starting value for the epsilon greedy choice of actions in the deep models')
    # parameters for RL
    parser.add_argument('--gamma', type=float, default=0.99, help='the discount value')
    parser.add_argument('--epsilon', type=float, default=0.2, help='the epsilon value for the epsilon greedy choice of actions (this parameter does not apply to deep models)')
    parser.add_argument('--alpha', type=float, default=0.5, help='Alpha therm for SARSA and Q-learning')
    parser.add_argument('--lambda_s', type=float, default=0.2, help='lambda therm for SARSA-lambda')
    

    # Parse arguments
    args = parser.parse_args()

    # Retrieve arguments
    save_path = args.save_path

    run_all = args.run_all
    run_AC = args.run_AC
    run_DQN = args.run_DQN
    run_VI = args.run_VI
    run_SA = args.run_SA
    run_SAL = args.run_SAL
    run_Q = args.run_Q
    run_RA = args.run_RA

    path_to_file_maze = args.maze_path
    maze_name = args.maze_name
    maze_x = args.maze_x
    maze_y = args.maze_y
    maze_fp = args.maze_fp
        
    batch_size = args.batch_size
    epsilon_decay = args.epsilon_decay
    epsilon_start = args.epsilon_start
    episodes = args.episodes
    replay_buffer_size = args.replay_buffer_size
    cuda = args.cuda
    
    discount_value = args.gamma
    epsilon = args.epsilon
    alpha = args.alpha
    lambda_s = args.lambda_s 
  
    actions = ["up","down","right","left"]

    if path_to_file_maze == "":
        #create new mazemaze_fp = args.maze_fp    
        maze = maze_generator().generate_maze(size_x=maze_x, size_y =maze_y, start_coord = [0,1], finish_coord = [maze_x-1,maze_y-1], n_of_turns = maze_fp)
        if maze_name == "":
            maze_name = (f"maze_{maze_x}-{maze_y}-{maze_fp}")
        
        path_to_file_maze=save_path+"/"+maze_name
        maze_generator().save_maze_as_csv(maze,path_to_file_maze)

    if run_VI or run_AC or run_DQN or run_Q or run_RA or run_SA or run_SAL:
        run_all = False

    if run_all:
        
        mean_reward, mean_steps, win_number = baseline.baseline(maze_path=path_to_file_maze, actions=actions, num_iterations=episodes).solve_maze()
        
        policy, V = dynamic_programming.value_iteration_class(discount_value=discount_value, maze_path=path_to_file_maze, actions=actions, num_iterations=episodes).value_iteration()
        val_iter_score = utils.common_functions().play_episode(policy, path_to_file_maze, show_window = False)

        q, policy = temporal_difference.sarsa_class(alpha = alpha, discount_value = discount_value, epsilon = epsilon, maze_path=path_to_file_maze, actions=actions, num_iterations = episodes).sarsa()
        sarsa_score = utils.common_functions().play_episode(policy, path_to_file_maze, step_by_step = False, show_window=False)
            
        q, policy = temporal_difference.sarsa_class(alpha = alpha, discount_value = discount_value, epsilon = epsilon, maze_path=path_to_file_maze, actions=actions, num_iterations = episodes).sarsa_lambda(lambda_par = lambda_s)
        sarsa_lambda_score = utils.common_functions().play_episode(policy, path_to_file_maze, show_window=False)

        q, policy = temporal_difference.Q_learning_class(alpha = alpha, discount_value=discount_value, maze_path=path_to_file_maze, actions=actions).Q_learning()
        Q_learning_score = utils.common_functions().play_episode(policy, path_to_file_maze, show_window=False)

        dqn_score = DeepQNetwork.train_dqn(episodes=episodes, path_to_file_maze = path_to_file_maze, batch_size = batch_size, cuda = cuda, epsilon_start=epsilon_start, epsilon_decay = epsilon_decay, replay_buffer_size=replay_buffer_size)

        a2c_score = actor_critic.actor_critic_class(gamma = discount_value, maze_path = path_to_file_maze, actions= actions, max_episodes = episodes, epsilon_start=epsilon_start, epsilon_decay=epsilon_decay, cuda = cuda).update()

        print("Value of random steps:  mean reward ", str(mean_reward), " mean steps ", str(mean_steps), " Wins: ", str(win_number), "/",episodes)

        print("Value iteration solved the maze with a score of: ", str(val_iter_score))
        
        print("Q_learning solved the maze with a score of: ", str(Q_learning_score))
        print("SARSA iteration solved the maze with a score of: ", str(sarsa_score))
        print("SARSA-lambda iteration solved the maze with a score of: ", str(sarsa_lambda_score))

        print("DQN  solved the maze with a score of: ", str(dqn_score))
        print(" (with experience replay and epsilon decay)")
        print("Actor Critic solved the maze with a score of: ", str(a2c_score))
        print(" (with epsilon decay)")
        
    else:
        if run_VI:
            policy, V = dynamic_programming.value_iteration_class(discount_value=discount_value, maze_path=path_to_file_maze, actions=actions, num_iterations=episodes).value_iteration()
            val_iter_score = utils.common_functions().play_episode(policy, path_to_file_maze, show_window = False)
            print("Value iteration solved the maze with a score of: ", str(val_iter_score))
        
        if run_SA:
            q, policy = temporal_difference.sarsa_class(alpha = alpha, discount_value = discount_value, epsilon = epsilon, maze_path=path_to_file_maze, actions=actions, num_iterations = episodes).sarsa()
            sarsa_score = utils.common_functions().play_episode(policy, path_to_file_maze, step_by_step = False, show_window=False)
            print("SARSA iteration solved the maze with a score of: ", str(sarsa_score))
            
        if run_SAL:
            q, policy = temporal_difference.sarsa_class(alpha = alpha, discount_value = discount_value, epsilon = epsilon, maze_path=path_to_file_maze, actions=actions, num_iterations = episodes).sarsa_lambda(lambda_par = lambda_s)
            sarsa_lambda_score = utils.common_functions().play_episode(policy, path_to_file_maze, show_window=False)
            print("SARSA-lambda iteration solved the maze with a score of: ", str(sarsa_lambda_score))

        if run_Q:
            q, policy = temporal_difference.Q_learning_class(alpha = alpha, discount_value=discount_value, maze_path=path_to_file_maze, actions=actions).Q_learning()
            Q_learning_score = utils.common_functions().play_episode(policy, path_to_file_maze, show_window=False)
            print("Q_learning solved the maze with a score of: ", str(Q_learning_score))

        if run_DQN:
            dqn_score = DeepQNetwork.train_dqn(episodes=episodes, path_to_file_maze = path_to_file_maze, batch_size = batch_size, cuda = cuda, epsilon_start=epsilon_start, epsilon_decay = epsilon_decay, replay_buffer_size=replay_buffer_size)
            print("DQN  solved the maze with a score of: ", str(dqn_score))
            print(" (with experience replay and epsilon decay)")
            
        if run_AC:
            a2c_score = actor_critic.actor_critic_class(gamma = discount_value, maze_path = path_to_file_maze, actions= actions, max_episodes = episodes, epsilon_start=epsilon_start, epsilon_decay=epsilon_decay, cuda = cuda).update()
            print("Actor Critic solved the maze with a score of: ", str(a2c_score))
            print(" (with epsilon decay)")
        
        if run_RA:
            mean_reward, mean_steps, win_number = baseline.baseline(maze_path=path_to_file_maze, actions=actions, num_iterations=episodes).solve_maze()
            print("Value of random steps:  mean reward ", str(mean_reward), " mean steps ", str(mean_steps), " Wins: ", str(win_number), "/",episodes)
            
       
main()

"""
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
        python3 ./main.py --maze_path ./saved_maze/maze4  --run_DQN True --episodes 2000 --gamma 0.99 --epsilon_start 1 --epsilon_decay 0.999 --cuda True 
    
"""
