import numpy as np
from toolbox import N, S, E, W, discreteProb
from maze_plotter import maze_plotter # used to plot the maze

    
class simple_actspace(): #class describing the action space of the markov decision process
    def __init__(self, action_list=[], nactions=0):
        if len(action_list) == 0:
            self.actions = np.array([a for a in range(nactions)])
        else:
            self.actions = action_list
            
        self.size = len(self.actions)
        
    def sample(self, prob_list=None): #returns an action drawn according to the prob_list distribution, 
        # if the param is not set, then it is drawn from a uniform distribution 
        if prob_list is None :
            prob_list = np.ones((self.size))/self.size
            
        index = discreteProb(prob_list) 
        return  self.actions[index]
    

        
    
class mdp(): #defines a Markov Decision Process

    def __init__(self, observation_space, action_space, start_distribution, transition_matrix,
                  reward_matrix, plotter, gamma=0.9, terminal_states=[], timeout=50):
        
        self.observation_space = observation_space
        self.terminal_states = terminal_states
        self.action_space = action_space
        self.current_state = -1 #current position of the agent in the maze, it's set by the method reset()
        self.timeout = timeout #maximum length of an episode
        self.timestep = 0 
        self.P0 = start_distribution #distribution used to draw the first state of the agent, used in method reset()
        self.P = transition_matrix
        self.r = reward_matrix
        self.plotter = plotter #used to plot the maze
        self.gamma = gamma #discount factor
        self.last_action_achieved = False #used to tell whether the last state has been reached or not (see done())
    
    

    def reset(self, uniform=False): #initializes an episode and returns the state of the agent
        #if uniform is set to False, the first state is drawn according to the P0 distribution, 
        #else it's drawn on a uniform distribution over all the states
        
        if uniform :
            prob = np.ones((self.observation_space.size))/self.observation_space.size
            self.current_state = discreteProb(prob)
        else :
            self.current_state = discreteProb(self.P0)
            
        self.timestep = 0
        self.last_action_achieved = False
        
        return self.current_state
 
    
    def step(self,u,deviation=0): # performs a step forward in the environment, 
        # if you want to add some noise to the reward, give a value to the deviation param 
        # which represents the mean μ of the normal distribution used to draw the noise 
        
        noise = 0 # = deviation*np.random.randn() # generate noise, see an exercize in mbrl.ipynb
        reward = self.r[self.current_state,u] +noise # r is the reward of the transition, you can add some noise to it 
        
        # the state reached when performing action u from state x is sampled 
        # according to the discrete distribution self.P[x,u,:]
        observation = discreteProb(self.P[self.current_state,u,:]) 
        
        self.timestep += 1 
        
        
        info = {} #can be used when debugging
        info["State transition probabilities"] = self.P[self.current_state,u,:]
        info["reward's noise value"] = noise
        
        self.current_state = observation
        done = self.done() #checks if the episode is over
        
        return [observation,reward,done,info]
    
    
    def done(self): #returns True if the episode is over
        if self.last_action_achieved :
            return True
        if self.current_state in self.terminal_states: #done when a terminal state is reached
            #the terminal states are actually a set of states from which any action leads to an added imaginary state, 
            #the "well", with a reward of 1. To know if the episode is over, we have to check
            #whether the agent is on one of these last states and performed the action that gives it its last reward 
            self.last_action_achieved = True
            
        return self.timestep == self.timeout #done when timeout reached
    
    
    def new_render(self): #initializes a new environment rendering (a plot defined by a figure, an axis...)
        self.plotter.new_render()
    
    def render(self, V=[], policy=[], agent_pos=-1): #outputs the agent in the environment with values V (or Q)
        
        if agent_pos > -1:
            self.plotter.render(agent_state=agent_pos, V=V, policy=policy)
        elif self.current_state > -1:# and not self.last_action_achieved:
            self.plotter.render(agent_state=self.current_state, V=V, policy=policy)
        else :
            self.plotter.render(V=V, policy=policy)
        
    def save_fig(self, title): #saves the current output into the disk
        self.plotter.save_fig(title)
            
    def create_animation(self,V_list=[],policy_list=[],nframes=0): #given a list of V or Q values, a list of policies, 
        # and eventually the number of frames wanted, it generates a video of the different steps
        return self.plotter.create_animation(V_list,policy_list,nframes)
    

class maze(): #describes a maze-like environment
    def __init__(self, width, height, walls=[]):
        self.width = width
        self.height = height
        self.states = np.array([s for s in range(width*height)])
        self.walls = walls
        self.size = width*height
     

    
class maze_mdp(mdp):  # defines a Markov Decision Process which observation space is a maze

    def __init__(self, width, height, walls=[], action_list=[], nactions=4,
                 gamma=0.9, timeout=50, start_states=[0], terminal_states=[], rewards=[], stochasticity=0):
        # width, height : int numbers defining the maze attributes
        # walls : list of the states that represent walls in our maze environment
        # action_list : list of possible actions
        # nactions : used when action_list is empty, by default there are 4 of them (go north, south, eat or west)
        # gamma : the discount factor of our mdp
        # timeout : defines the length of an episode (max timestep) --see done() function
        # start_states : list that defines the states where the agent can be at the beginning of an episode
        # terminal_states : list that defines the states corresponding to the end of an episode
        #                  (agent reaches a terminal state) --cf. done() function
        
        # ##################### State Space ######################
        
        observation_space = maze(width, height, walls)
        
        # ##################### Action Space ######################
        
        action_space = simple_actspace(action_list=action_list, nactions=nactions)    

        # ##################### Distribution Over Initial States ######################
        
        start_distribution = np.zeros(observation_space.size)  # distribution over initial states
        
        for state in start_states:
            start_distribution[state] = 1.0/len(start_states)

        # ##################### Transition Matrix ######################
        
        transition_matrix = np.empty((observation_space.size+1, action_space.size, observation_space.size+1))  # a "well" state is added that only the terminal states can get into

        # Transition Matrix when going north
        transition_matrix[:, N, :] = np.zeros((observation_space.size+1, observation_space.size+1))
        for i in observation_space.states:
            north = False
            south = False
            east = False
            west = False
            if i == 0 or i % observation_space.height == 0 or i-1 in observation_space.walls or i in observation_space.walls:  # the state doesn't change (highest cells + cells under a wall)
                transition_matrix[:, N, :][i][i] = 1.0
            else:  # it goes up
                transition_matrix[:, N, :][i][i-1] = 1.0
                n_path=0
                if not(i % observation_space.height == observation_space.height-1 or i+1 in observation_space.walls or i in observation_space.walls):
                    south = True
                    n_path+=1
                if not(i>observation_space.size-observation_space.height-1 or i+observation_space.height in observation_space.walls or i in observation_space.walls):
                    east = True
                    n_path+=1
                if not(i < observation_space.height or i-observation_space.height in observation_space.walls or i in observation_space.walls):
                    west = True
                    n_path+=1
                if south:
                    transition_matrix[:, N, :][i][i-1] -= stochasticity/n_path
                    transition_matrix[:, N, :][i][i+1] = stochasticity/n_path
                if east:
                    transition_matrix[:, N, :][i][i-1] -= stochasticity/n_path
                    transition_matrix[:, N, :][i][i+height] = stochasticity/n_path
                if west:
                    transition_matrix[:, N, :][i][i-1] -= stochasticity/n_path
                    transition_matrix[:, N, :][i][i-height] = stochasticity/n_path
                        

        # Transition Matrix when going south
        transition_matrix[:, S, :] = np.zeros((observation_space.size+1, observation_space.size+1))
        for i in observation_space.states:
            north = False
            south = False
            east = False
            west = False
            if i % observation_space.height == observation_space.height-1 or i+1 in observation_space.walls or i in observation_space.walls:  # the state doesn't change (lowest cells + cells above a wall)
                transition_matrix[:, S, :][i][i] = 1.0
            else:  # it goes down
                transition_matrix[:, S, :][i][i+1] = 1.0
                n_path=0
                if not(i == 0 or i % observation_space.height == 0 or i-1 in observation_space.walls or i in observation_space.walls):
                    north = True
                    n_path+=1
                if not (i>observation_space.size-observation_space.height-1 or i+observation_space.height in observation_space.walls or i in observation_space.walls):
                    east = True
                    n_path+=1
                if not (i < observation_space.height or i-observation_space.height in observation_space.walls or i in observation_space.walls):
                    west = True
                    n_path+=1
                if north:
                    transition_matrix[:, S, :][i][i+1] -= stochasticity/n_path
                    transition_matrix[:, S, :][i][i-1] = stochasticity/n_path
                if east:
                    transition_matrix[:, S, :][i][i+1] -= stochasticity/n_path
                    transition_matrix[:, S, :][i][i+height] = stochasticity/n_path
                if west:
                    transition_matrix[:, S, :][i][i+1] -= stochasticity/n_path
                    transition_matrix[:, S, :][i][i-height] = stochasticity/n_path


    
        # self.P[:,S,:][49][50] = 0.2 #example for hacking local probabilities
        # self.P[:,S,:][49][48] = 0.8

        # Transition Matrix when going east
        transition_matrix[:, E, :] = np.zeros((observation_space.size+1, observation_space.size+1))
        for i in observation_space.states:
            north = False
            south = False
            east = False
            west = False
            if i < observation_space.height or i-observation_space.height in observation_space.walls or i in observation_space.walls:  # state doesn't change (cells on the right side of a wall)
                transition_matrix[:, E, :][i][i] = 1.0
            else:  # it goes left
                transition_matrix[:, E, :][i][i-height] = 1.0
                n_path=0
                if not (i == 0 or i % observation_space.height == 0 or i-1 in observation_space.walls or i in observation_space.walls):
                    north = True
                    n_path+=1
                if not (i % observation_space.height == observation_space.height-1 or i+1 in observation_space.walls or i in observation_space.walls):
                    south=True
                    n_path+=1
                if not(i>observation_space.size-observation_space.height-1 or i+observation_space.height in observation_space.walls or i in observation_space.walls):
                    west = True
                    n_path+=1
                if north:
                    transition_matrix[:, E, :][i][i-height] -= stochasticity/n_path
                    transition_matrix[:, E, :][i][i-1] = stochasticity/n_path
                if south:
                    transition_matrix[:, E, :][i][i-height] -= stochasticity/n_path
                    transition_matrix[:, E, :][i][i+1] = stochasticity/n_path
                if west:
                    transition_matrix[:, E, :][i][i-height] -= stochasticity/n_path
                    transition_matrix[:, E, :][i][i+height] = stochasticity/n_path


        # Transition Matrix when going west
        transition_matrix[:, W, :] = np.zeros((observation_space.size+1, observation_space.size+1))
        for i in observation_space.states:
            north = False
            south = False
            east = False
            west = False
            if i>observation_space.size-observation_space.height-1 or i+observation_space.height in observation_space.walls or i in observation_space.walls: #state doesn't change (cells on the left side of a wall)
                transition_matrix[:, W, :][i][i] = 1.0
            else:  # it goes right
                transition_matrix[:, W, :][i][i+height] = 1.0
                n_path=0
                if not (i == 0 or i % observation_space.height == 0 or i-1 in observation_space.walls or i in observation_space.walls):
                    north = True
                    n_path+=1
                if not (i % observation_space.height == observation_space.height-1 or i+1 in observation_space.walls or i in observation_space.walls):
                    south=True
                    n_path+=1
                if not(i < observation_space.height or i-observation_space.height in observation_space.walls or i in observation_space.walls):
                    east=True
                    n_path+=1
                if north:
                    transition_matrix[:, W, :][i][i+height] -= stochasticity/n_path
                    transition_matrix[:, W, :][i][i-1] = stochasticity/n_path
                if south:
                    transition_matrix[:, W, :][i][i+height] -= stochasticity/n_path
                    transition_matrix[:, W, :][i][i+1] = stochasticity/n_path
                if east:
                    transition_matrix[:, W, :][i][i+height] -= stochasticity/n_path
                    transition_matrix[:, W, :][i][i-height] = stochasticity/n_path

                
        # Transition Matrix of final states 
        well = observation_space.size  # all the final states' transitions go there
        for s in terminal_states:
            transition_matrix[s, :, :] = 0
            transition_matrix[s, :, well] = 1

        # for i in observation_space.states:
        #     for j in range(4):
        #         for k in observation_space.states:
        #             print (i,j,k,"::",transition_matrix[i][j][k])
        # Transition Matrix when not moving (action removed from the current version)
        # transition_matrix[:,NoOp,:] = np.eye(observation_space.size)

        # ##################### Reward Matrix ######################

        if rewards == []:
            rewards = terminal_states

        reward_matrix = np.zeros((observation_space.size, action_space.size)) 
        for s in rewards:
            reward_matrix[s, :] = 1  # leaving a final state gets the agent a reward of 1
        # reward_matrix[-1][NoOp] = 1.0
        # reward_matrix[25][NoOp] = 0.9
        
        plotter = maze_plotter(observation_space, terminal_states)  # renders the environment
        mdp.__init__(self, observation_space, action_space, start_distribution, transition_matrix,
                 reward_matrix, plotter, gamma=gamma, terminal_states=terminal_states, timeout=timeout)

    def reset(self, uniform=False):  # initializes an episode
        # if uniform is set to False, the first state is drawn from the P0 distribution,
        # else it is drawn from a uniform distribution over all the states except for walls
        if uniform:
            prob = np.ones(self.observation_space.size)/(self.observation_space.size-len(self.observation_space.walls))
            for state in self.observation_space.walls:
                prob[state] = 0.0
            self.current_state = discreteProb(prob)
        else:
            self.current_state = discreteProb(self.P0)

        self.timestep = 0
        self.last_action_achieved = False
        return self.current_state

        