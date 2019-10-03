import numpy as np

##########-action constants-###################
N = 0
S = 1
E = 2
W = 3
NOOP = 4  


def discreteProb(p):
        # Draw a random number using probability table p (column vector)
        # Suppose probabilities p=[p(1) ... p(n)] for the values [1:n] are given, sum(p)=1 
        # and the components p(j) are nonnegative. 
        # To generate a random sample of size m from this distribution,
        #imagine that the interval (0,1) is divided into intervals with the lengths p(1),...,p(n). 
        # Generate a uniform number rand, if this number falls in the jth interval given the discrete distribution,
        # return the value j. Repeat m times.
        r = np.random.random()
        cumprob=np.hstack((np.zeros(1),p.cumsum()))
        sample = -1
        for j in range(p.size):
            if (r>cumprob[j]) & (r<=cumprob[j+1]):
                sample = j
                break
        return sample

def softmax(Q,x,tau):
    # Returns a soft-max probability distribution over actions
    # Inputs :
    # - Q : a Q-function represented as a nX times nU matrix
    # - x : the state for which we want the soft-max distribution
    # - tau : temperature parameter of the soft-max distribution
    # Output :
    # - p : probability of each action according to the soft-max distribution
    
    p = np.zeros((len(Q[x])))
    sump = 0
    for i in range(len(p)) :
        p[i] = np.exp((Q[x,i]/tau).round(5))
        sump += p[i]
    
    p = p/sump
    
    return p



def compare(V,Q,pol): 
    # compares the state value V with the state-action value Q following policy pol
    epsilon = 0.01 # precision of the comparison
    sumval = np.zeros((V.size))
    for i in range(V.size): # compute the difference between V and Q for each state
        sumval[i] = abs(V[i] - Q[i,pol[i]])
         
    if np.max(sumval)<epsilon :
        return True
    else :
        return False

