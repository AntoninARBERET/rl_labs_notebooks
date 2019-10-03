import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from matplotlib.table import Table
from matplotlib import rc
from ipynb.fs.defs.toolbox import N,S,E,W


###################-plot functions for a maze like environment-#################


##################################################

#### maze_mdp plot, used to plot the agent in its environment while processing the V/Q function and policy 
#### it can also create videos given a list of V/Q values and a list of policies

class maze_plotter():
    def __init__(self, maze, terminal_states): # maze defined in the mdp notebook
        self.maze_attr = maze
        self.terminal_states = terminal_states
        plt.ion()
        self.figW = self.maze_attr.width
        self.figH = self.maze_attr.height
        self.figure_history=[]
        self.axes_history=[]
        self.table_history=[]
        self.agent_patch_history = []
        
    def init_table(self): # the states of the mdp are drawn in a matplotlib table, this function creates this table
        
        width = 1 #0.1
        height = 1 #0.2
        
        for i in range(self.maze_attr.height) :
            for j in range(self.maze_attr.width) :
                state = j*self.maze_attr.height + i
                color = np.zeros(3)
                if state in self.maze_attr.walls :
                    color[0]=color[1]=color[2]=0
                else :
                    color[0]=color[1]=color[2]=1
                self.table_history[-1].add_cell(i,j, width, height, facecolor=color, text='', loc='center')
        
        self.axes_history[-1].add_table(self.table_history[-1])
    
    def new_render(self): # initializes the plot by creating its basic components (figure, axis, agent patch and table)
        # a trace of these components is stored so that the old outputs will last on the notebook 
        # when a new rendering is performed
        self.figure_history.append(plt.figure(figsize=(self.figW,self.figH)))
        self.axes_history.append(self.figure_history[-1].add_subplot(111)) # 111 : number of rows, columns and index of subplot
        self.table_history.append(Table(self.axes_history[-1], bbox=[0,0,1,1])) # [left, bottom, width, height]
        self.agent_patch_history.append(mpatches.Ellipse((-1,-1), 0.06, 0.085, ec="none", fc="dodgerblue", alpha=0.6))
        self.axes_history[-1].add_patch(self.agent_patch_history[-1])
        self.init_table()

    def coords(self, height, width, state): #processes the starting position of the arrows
        i = state%self.maze_attr.height
        j = int(state/self.maze_attr.height)
        h = 1/self.figH
        ch = h/2
        w = 1/self.figW
        cw = w/2
        x,y = j*w + cw,1-(i*h + ch)
        return x,y
    
    def render(self, agent_state=-1, V=[], policy=[], render=True): # updates the values of the table 
        # and the agent position and current policy 
        # some of these components may not show depending on the parameters given when calling this function
        if len(self.figure_history) == 0 : # new plot
            self.new_render()
        
       
        self.axes_history[-1].clear()
        self.axes_history[-1].add_table(self.table_history[-1])
        
        #### Table values and policy update
        if len(V)>0: # working with state values
            if len(V.shape)==1 :
                self.V_render(agent_state, V, policy, render) 
            else : # working with state values
                self.Q_render(agent_state, V, policy, render)

        
        if agent_state >= 0:
            x,y = self.coords(self.maze_attr.height, self.maze_attr.width, agent_state)
            
            
            self.agent_patch_history[-1].center = x,y
    
            self.axes_history[-1].add_patch(self.agent_patch_history[-1])
            #print(agent_state,i,x,j,y)
        
        #plt.subplots_adjust(left=0.2, bottom=0.2)
        plt.xticks([])
        plt.yticks([])
        if render :
            self.figure_history[-1].canvas.draw()
            self.figure_history[-1].canvas.flush_events()
        return self.figure_history[-1]
    
    
    def V_render(self, agent_state, V, policy, render):
        
        for i in range(self.maze_attr.height) :
            for j in range(self.maze_attr.width):
                state = j*self.maze_attr.height + i
                color = np.zeros(3)
                if state in self.maze_attr.walls:
                    color[0]=color[1]=color[2]=0
                else:
                    color[0]=color[1]=color[2]=np.min([1-V[state]/(np.max(V)+1),1])
                    
                self.table_history[-1]._cells[(i,j)].set_facecolor(color)

                self.table_history[-1]._cells[(i,j)]._text.set_text(np.round(V[state],2))
                
                if len(policy)>0 and not (state in self.maze_attr.walls or state in self.terminal_states):
                    x0, y0, x,y = self.arrow_params(self.maze_attr.height, self.maze_attr.width,
                                                        state, policy[state])
                    arw_color = "red"
                    alpha = 0.6
                        
                    if not(x == y and x == 0):
                        self.axes_history[-1].arrow(x0, y0, x, y, alpha=alpha,
                                      head_width=0.03, head_length=0.03, fc=arw_color, ec=arw_color)
        
        
    def Q_render(self, agent_state, Q, policy, render):
         
        for i in range(self.maze_attr.height) :
            for j in range(self.maze_attr.width):
                
                state = j*self.maze_attr.height + i
                color = np.zeros(3)
                if state in self.maze_attr.walls:
                    color[0]=color[1]=color[2]=0
                else:
                    color[0]=color[1]=color[2]=np.min([1-np.max(Q[state])/(np.max(Q)+1),1])
                    
                self.table_history[-1]._cells[(i,j)].set_facecolor(color)

                self.table_history[-1]._cells[(i,j)]._text.set_text(np.round(np.max(Q[state]),2))
                
                if not (state in self.maze_attr.walls or state in self.terminal_states):
                    qmin = np.min(Q[state])
                    if qmin < 0:
                        qmin *= -1
                    pos_Q = Q[state] + qmin
                    qmax = np.max(pos_Q)
                    norm_Q = pos_Q / (np.sum(pos_Q)-(list(pos_Q).count(qmax)*qmax)+0.1)
                    
                    
                    for action in range(len(Q[state])):
                        
                        x0, y0, x, y = self.qarrow_params(self.maze_attr.height, 
                                                    self.maze_attr.width, state, action)


                        arw_color = "green"
                        alpha = 0.9
                        qmax = np.max(Q[state])

                        if not Q[state][action]==qmax:
                            arw_color = "red"
                            alpha = norm_Q[action]

                        if x == 0 and y == 0:
                            circle = mpatches.Circle((x0, y0), 0.02, ec=arw_color, fc=arw_color, alpha=alpha)
                            self.axes_history[-1].add_patch(circle)
                        else:
                            self.axes_history[-1].arrow(x0, y0, x, y, alpha=alpha,
                                          head_width=0.03, head_length=0.02, fc=arw_color, ec=arw_color)



           
    def arrow_params(self, height, width, state, action): #processes the starting position of the arrows
        x,y= self.coords(height, width, state)
        
        if action == N :
            return [x, y+0.02, 0.0, 0.04]
        elif action == S :
            return [x, y-0.02, 0.0, -0.04]
        elif action == W :
            return [x-0.03, y, -0.02, 0.0]
        elif action == E :
            return [x+0.03, y, 0.02, 0.0]
        else :
            return [x, y, 0.0, 0.0]   
    
    def qarrow_params(self, height, width, state, action): #processes the starting position of the arrows
        x,y= self.coords(height, width, state)
        
        if action == N :
            return [x, y+0.03, 0.0, 0.0125] #1/(10*self.figH)]
        elif action == S :
            return [x, y-0.03, 0.0, -1/(10*self.figH)]
        elif action == W :
            return [x-0.03, y, -1/(10*self.figW), 0.0]
        elif action == E :
            return [x+0.03, y, 1/(10*self.figW), 0.0]
        else :
            return [x, y, 0.0, 0.0]   
        
        
    def save_fig(self, title):
        self.figure_history[-1].savefig(title)
        
    def update(self, frame, V_list, pol_list):
        if len(pol_list)>frame:
            return self.render(V=V_list[frame],policy=pol_list[frame], render=False)
        else:
            return self.render(V=V_list[frame], render=False)
    
    def create_animation(self, Q_list=[], pol_list=[], nframes=0):
        new_Qlist = Q_list
        new_polist = pol_list
        if nframes > 0 :
            new_Qlist, new_polist = self.resize_lists(Q_list, pol_list, nframes)
            
        self.new_render()
        anim = animation.FuncAnimation(self.figure_history[-1], self.update, frames=len(new_Qlist), 
                                       fargs=[new_Qlist, new_polist], blit=True)
        #plt.close()
        return anim
    
    def resize_lists(self, Q_list, policy_list, nb_frames): #gets samples from the data list to fit the number of frames 
        # to show in the animation
        # used when the length of the data list exceeds the number of the frames required for the video
                    
        if nb_frames < len(Q_list) :
            step = np.int(np.round(len(Q_list)/nb_frames,0))
            print("sample length : ",len(Q_list))
            print("nb of frames : ",nb_frames)
            print("step size : ",step)

            new_Qlist = []
            new_polist = []

            for i in range(0,len(Q_list),step) :
                new_Qlist.append(Q_list[i])
                if len(policy_list) > i:
                    new_polist.append(policy_list[i])            

            return new_Qlist, new_polist
        else :
            return Q_list, policy_list