import numpy as np

########################### Value Iteration ###########################
# Given a MDP, this algorithm computes the optimal state value function V
# It then derives the optimal policy based on this function

def VI_V(mdp, render=True): #Value Iteration using the state value V
    V = np.zeros((mdp.observation_space.size)) #initial state values are set to 0
    pol = np.zeros((mdp.observation_space.size)) #initial policy set to always go north
    quitt = False

    V_list = [V.copy()] #list of the state values computed over time (used to generate an animation)
    policy_list = [pol.copy()] #list of the policies computed over time (used to generate an animation)

    if render:
        mdp.new_render()

    while quitt==False:
        Vold = V.copy()
        if render:
            mdp.render(V, pol)

        for x in mdp.observation_space.states : #for each state x

            # Compute the value of the state x for each action u of the MDP action space
            V_temp = [] 
            for u in mdp.action_space.actions : 
                if not x in mdp.terminal_states:
                    # Process sum of the values of the neighbouring states
                    sum = 0
                    for y in mdp.observation_space.states:
                        sum = sum + mdp.P[x,u,y]*Vold[y]
                    V_temp.append(mdp.r[x,u] + mdp.gamma*sum) 
                else : # if the it is one of the final states, then we only take the rewardinto account
                    V_temp.append(mdp.r[x,u]) 

            # Select the highest state value among those computed
            V[x] = np.max(V_temp)

            # Set the policy for this state as the action u that maximizes the state value of x
            pol[x] = np.argmax(V_temp)

        V_list.append(V.copy())
        policy_list.append(pol.copy())

        # Test if convergence has been reached
        if (np.linalg.norm(V-Vold)) < 0.01 :
            quitt = True

    if render:
            mdp.render(V, pol)


    return [V_list, policy_list]


def VI_Q(mdp, render=True): #Value Iteration based on the state-action value Q
    #Same as above, but we save all the computed values in the Q table 
    #(instead of saving only the optimal value of each state), so there is no need for a V_temp list
    Q = np.zeros((mdp.observation_space.size,mdp.action_space.size))
    pol = np.zeros((mdp.observation_space.size))
    quitt = False

    Q_list = [Q.copy()] 
    pol_list = [pol.copy()]
    Qmax = Q.max(axis=1)

    if render:
        mdp.new_render()

    i=0
    while quitt==False:
        Qold = Q.copy()

        if render:
            mdp.render(Q,pol)
        i+=1
        #print("iteration : {}".format(i))
        for x in mdp.observation_space.states :
           # print("x = {}".format(x))
            for u in mdp.action_space.actions :
                #print("u = {}".format(u))
                sum =0
                for y in mdp.observation_space.states:
                    #if i==3 and x>51:
                        #print("y = {}".format(y))
                    uy_opt=0
                    uy_val=0
                    for uy in mdp.action_space.actions :
                    #if i == 3 and x>51:
                           # print("uy = {}".format(uy))
                        if(Qold[y,uy] > uy_val):
                            uy_val = Qold[y,uy]
                            uy_opt = uy

                    sum = sum + mdp.P[x,u,y]*Qold[y,uy_opt]


                Q[x,u] = mdp.r[x,u]+mdp.gamma*sum

        Qmax = Q.max(axis=1)
        pol =  np.argmax(Q,axis=1)

        Q_list.append(Q.copy())
        pol_list.append(pol)


        if (np.linalg.norm(Q-Qold)) <= 0.01 :
            quitt = True

    if render:
        mdp.render(Q,pol)

    return [Q_list, pol_list]



def PI_V(mdp, render=True):

    V = np.zeros((len(mdp.observation_space.states)))
    P = np.zeros((len(mdp.observation_space.states)),dtype=np.int16)
    V_list = [V.copy()]
    P_list = [P.copy()]

    quitt_policy = False
    if render:
        mdp.new_render()

    while quitt_policy==False:
        Pold = P.copy()
        quitt_value = False
        if render:
            mdp.render(V, P)

        while quitt_value==False:
            Vold = V.copy()
            for x in mdp.observation_space.states:
                u = Pold[x]
                if not x in mdp.terminal_states:
                    # Process sum of the values of the neighbouring states
                    sum = 0
                    for y in mdp.observation_space.states:
                        sum = sum + mdp.P[x,u,y]*Vold[y]
                    V[x] = mdp.r[x,u] + mdp.gamma*sum
                else : # if the it is one of the final states, then we only take the rewardinto account
                    V[x] = mdp.r[x,u]

            if (np.linalg.norm(V-Vold)) <= 0.01 :
                quitt_value = True

        for x in mdp.observation_space.states:
            if not x in mdp.terminal_states:
                P_x = dict()
                for u in mdp.action_space.actions:
                    sum = 0
                    for y in mdp.observation_space.states:
                        sum = sum + mdp.P[x,u,y]*V[y]
                    P_x[u]=sum
                P[x] = max(P_x, key=P_x.get) #get action with the best weighted values

        if (np.linalg.norm(P-Pold)) <= 0.01 :
            quitt_policy = True

        V_list.append(V.copy())
        P_list.append(P.copy())

    return V_list, P_list