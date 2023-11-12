import numpy as np
import random
import tqdm

from maze_generator import *
from minosses_maze import *
from maze_ui import *
from utils import *

class temporal_difference_class():
    
    def __init__(self, alpha = 0.4, discount_value = 0.99, epsilon = 0.1, load_maze = False, maze_args = [], actions= [], num_iterations = 1000, player_char = 9):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = discount_value
        self.actions = actions
        self.num_iteration = num_iterations
        self.player_char = player_char

        generator = maze_generator()
        

        if load_maze:
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
    
    def epsilon_greedy_policy(self, q, state, epsilon):
        """
            This method returns with epsilon probability either a random action or the best action
        """
        utils = common_functions()
        if random.uniform(0,1) < epsilon:
            return self.actions[random.randrange(0,len(self.actions))]
        else:            
            s = utils.find_index_of_coordinate(self.path, state)
            index_of_action = max(list(range(len(self.actions))), key = lambda x: q[(s,self.actions[x])])
            return self.actions[index_of_action]


    def build_policy_from_q(self, q):
        policy = np.zeros([len(self.path),len(self.actions)])/len(self.actions)
        
        for row in range(self.size_x):
                for col in range(self.size_y):   
                    state = [row, col]
                    if state in self.path:
                        action_values = []
                        s = common_functions().find_index_of_coordinate(self.path, state)
                        for action in self.actions:                            
                            state_action_enc = (s,action)
                            action_values.append(q[state_action_enc])
                        
                        best_action = np.argmax(action_values)
                        policy[s, best_action] = 1
        
        return policy

class Q_learning_class(temporal_difference_class):
    def Q_learning(self):
        q = {}
        for s in range(len(self.path)):
            for a in range(len(self.actions)):
                q[(s,self.actions[a])] = 0.0

        for i in tqdm(range(self.num_iteration), desc="Q-Learning learning: "):
            utils = common_functions(self.maze)
            r = 0  
            state = self.start_coord
            done = False

            while True:
                
                # In each state, we select the action by epsilon-greedy policy
                action = self.epsilon_greedy_policy(q, state, self.epsilon)
                # then we perform the action and move to the next state, and receive the reward
                next_state, reward = utils.action_value_function(state, action, self.finish_coord)
                if reward > 0:
                    done = True
                # Next we update the Q value using our update_q_table function
                # which updates the Q value by Q learning update rule
                
                q = self.update_q_table(q, state, action, reward, next_state, self.alpha, self.gamma)
                
                # Finally we update the previous state as next state
                state = next_state

                # Store all the rewards obtained
                r += reward

                #we will break the loop, if we are at the terminal state of the episode
                if done:
                    break
        policy = self.build_policy_from_q(q=q)

        common_functions().save_dictionary_to_csv(q, "./saved_maze/q_q_learning")
        maze_generator().save_maze_as_csv(policy, "./saved_maze/policy_q_learning", "float")

        return q, policy
        
    def update_q_table(self, q, state, action, reward, next_state, alpha, gamma):

        nxt_s = utils.find_index_of_coordinate(self.path, next_state)
        qa = max([q[(nxt_s, self.actions[a])] for a in range(len(self.actions))])

        s = utils.find_index_of_coordinate(self.path, state)
        q[(s,action)] += alpha * (reward + gamma * qa - q[(s,action)])

        return q
         

class sarsa_class(temporal_difference_class):
    def sarsa(self):
        Q = {}
        for s in range(len(self.path)):
            for a in range(len(self.actions)):
                Q[(s,self.actions[a])] = 0.0
        
        for i in tqdm(range(self.num_iteration), desc="SARSA learning: "):
    
            utils = common_functions(self.maze)
            # initialize the state,
            state = self.start_coord
            done = False
            
            # select the action using epsilon-greedy policy
            action = self.epsilon_greedy_policy(Q, state, self.epsilon)
            while True:
                # then we perform the action and move to the next state, and receive the reward
                next_state, reward = utils.action_value_function(state, action, self.finish_coord)
                if reward > 0:
                    done = True
                
                # again, we select the next action using epsilon greedy policy
                next_action = self.epsilon_greedy_policy(Q, next_state, self.epsilon)

                nxt_s = utils.find_index_of_coordinate(self.path, next_state)
                s = utils.find_index_of_coordinate(self.path, state)
                # we calculate the Q value of previous state using our update rule
                Q[(s,action)] += self.alpha * ((reward + self.gamma * Q[(nxt_s,next_action)])-Q[(s,action)])

                # finally we update our state and action with next action and next state
                action = next_action
                state = next_state
                
                # we will break the loop, if we are at the terminal state of the episode
                if done:
                    break
        
        policy = self.build_policy_from_q(q=Q)

        common_functions().save_dictionary_to_csv(Q, "./saved_maze/Q_sarsa")
        maze_generator().save_maze_as_csv(policy, "./saved_maze/policy_sarsa", "float")

        return Q, policy
    
    def sarsa_lambda(self, lambda_par = 3):
        Q = {}
        for s in range(len(self.path)):
            for a in range(len(self.actions)):
                Q[(s,self.actions[a])] = 0.0
        
        for i in tqdm(range(self.num_iteration), desc="SARSA-lambda learning: "):
            
            E = {}
            for s in range(len(self.path)):
                for a in range(len(self.actions)):
                    E[(s,self.actions[a])] = 0.0
            
            state = self.start_coord
            action = self.actions[1]#self.epsilon_greedy_policy(Q, state, self.epsilon)
            done = False
            
            while True:
                utils = common_functions(self.maze)
                
                next_state, reward = utils.action_value_function(state, action, self.finish_coord)
                if reward > 0:
                    done = True
                
                # again, we select the next action using epsilon greedy policy
                next_action = self.epsilon_greedy_policy(Q, next_state, self.epsilon)
                
                nxt_s = utils.find_index_of_coordinate(self.path, next_state)
                s = utils.find_index_of_coordinate(self.path, state)
                
                delta = reward + (self.gamma * Q[(nxt_s,next_action)])-Q[(s,action)]
                E[(s,action)] += 1
                
                for s in range(len(self.path)):
                    for a in range(len(self.actions)):
                        Q[(s,self.actions[a])] += self.alpha * delta * E[(s,self.actions[a])]
                        E[(s,self.actions[a])] = self.gamma * lambda_par * E[(s,self.actions[a])]

                
                state = next_state
                action = next_action

                if done: break
            
        policy = self.build_policy_from_q(q=Q)

        common_functions().save_dictionary_to_csv(Q, "./saved_maze/Q_sarsa_lambda")
        maze_generator().save_maze_as_csv(policy, "./saved_maze/policy_sarsa_lambda", "float")

        return Q, policy
    

actions = ["up","down","right","left","jump_up","jump_down","jump_right","jump_left"]
utils = common_functions()
path_to_file_maze = "./saved_maze/maze3"

#(0.5, 0.99, 0.9, 0, 27.30000000000002)
alphas = [0.5] #[0.5, 0.6, 0.99, 1]
gammas = [0.99] #[0.85, 0.9, 0.99]
epsilons = [0.2] #[0.8, 0.9, 0.99]
lambdas = [0.25] #[0.25, 0.35, 0.5, 0.75, 0.85, 0.9, 0.99]

results = []


c = 0
comb_numb = len(alphas) * len(gammas) * len(epsilons) * len(lambdas)

#for la in lambdas:
for alpha in alphas:
    for gamma in gammas:
        for epsilon in epsilons:       
            print("testing combination ", c, "/", comb_numb, "alpha: "+str(alpha)+" gamma: "+str(gamma)+" lambda: "+str(0) +" epsilon: "+str(epsilon))
            c+=1
            q, policy = sarsa_class(alpha = alpha, discount_value = gamma, epsilon = epsilon, load_maze=True, maze_args=[path_to_file_maze], actions=actions, num_iterations = 6000).sarsa()
            sarsa_score = utils.play_episode(policy, path_to_file_maze, step_by_step = False)
            results.append((alpha, gamma, epsilon, 0, sarsa_score))
            #maze_generator().save_maze_as_csv(policy, ("./saved_maze/lambda_sarsa_policy/policy_sarsa_lambda"+str(alpha)+"_"+str(gamma)+"_"+str(epsilon)+"_"+str(0)), "float")
            #common_functions().save_dictionary_to_csv(q, ("./saved_maze/lambda_sarsa_Q/Q_sarsa_lambda"+str(alpha)+"_"+str(gamma)+"_"+str(epsilon)+"_"+str(0)))


pos_result = []
for result in results:
    print(result)
    if result[4] > 0:
        pos_result.append(result)

rerun_score = []
for result in pos_result:
    print(result)
    q, policy = sarsa_class(alpha = result[0], discount_value = result[1], epsilon = result[2], load_maze=True, maze_args=[path_to_file_maze], actions=actions, num_iterations = 4000).sarsa()
    sarsa_lambda_score = utils.play_episode(policy, path_to_file_maze, step_by_step = False)
    rerun_score.append(sarsa_lambda_score)
print("---")
print(rerun_score)



q, policy = sarsa_class(alpha = 0.5, discount_value = 0.99, epsilon = 0.2, load_maze=True, maze_args=[path_to_file_maze], actions=actions, num_iterations = 2000).sarsa_lambda(lambda_par = 0.25)
sarsa_lambda_score = utils.play_episode(policy, path_to_file_maze)

q, policy = Q_learning_class(alpha = 0.6 ,discount_value=0.99, load_maze=True, maze_args=[path_to_file_maze], actions=actions).Q_learning()
Q_learning_score = utils.play_episode(policy, path_to_file_maze)


print("Q_learning solved the maze with a score of: ", str(Q_learning_score))
#print("SARSA iteration solved the maze with a score of: ", str(sarsa_score))
#print("SARSA-lambda iteration solved the maze with a score of: ", str(sarsa_lambda_score))