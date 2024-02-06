#Utility File for Virtual Reservoir Computer

#Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplot
import networkx as nx
import pickle
import random


#PREPROCESSING FUNCTIONS

def note_input_dict(W_inp):
    '''
    Returns look-up tables for note to Input node conversion
    
    Parameters:
    W_inp: (Numpy Array): Array of Input Weight Matrix
    '''
    notes = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
    dict_notes = {}
    #Add dictionary entries
    for i in range(7):
        dict_notes[notes[i]] = np.flatnonzero(W_inp)[i] #Assigns to every note the index of the corresponding node
        
    return dict_notes


#Convert to integer, One-Hot-Encoding (Matrix Encoding)
def note_to_matrix(X, dict_notes, len_matrix):
    '''
    Returns an Input Matrix for a sequence of notes
    
    Parameters: 
    X: (List) list containing sequences of notes to be transformed
    '''
    X_transcribed = []
    for sequence in X:
        M_new = np.zeros((len(sequence), len_matrix), dtype = bool) #Array of dimension (Reservoir nodes x sequence length)
        for i, note in enumerate(sequence):
            indice = dict_notes[note[0]]
            M_new[i, indice] = 1
        X_transcribed.append(M_new)
    return X_transcribed


#RESERVOIR COMPUTER FUNCTIONS

#RESERVOIR MATRIX

def ResMat(N, ConnProb, Spectral_radius):
    '''
    Creates a Reservoir Matrix
    
    Input:
    N: (Integer) number of nodes
    ConnProb: (float in [0,1]) probability of connection between two nodes
    Spectral_radius (float) rescales the network to desired spectral radius
    
    Output: 
    G: Networkx Erdos Renyi graph
    ResMat: Numpy array of the Reservoir Matrix
    '''
    G = nx.erdos_renyi_graph(N, ConnProb, seed=None, directed=True) #Generate random network
    
    ### Remove isolated nodes
    G.remove_nodes_from(list(nx.isolates(G)))
    N = G.number_of_nodes()
    
    ### Randomize weights between (0, 1)
    #Rand_weights = np.random.random((N,N))
    GNet = nx.to_numpy_array(G)
    #GNet = np.multiply(GNet,Rand_weights)
    
    ### Rescaling to a desired spectral radius 
    Spectral_radius_GNet = max(abs(np.linalg.eigvals(GNet)))
    ResMat = GNet*Spectral_radius/Spectral_radius_GNet
    
    return G, ResMat


def visualize_reservoir(G):
    '''
    Visualizes the Reservoir Matrix
    
    Input: 
    G: (Networks Graph) Networkz graph of the Reservoir
    '''
    pos=nx.spring_layout(G)
    
    #Give every node a number
    labels = {}
    for idx, node in enumerate(G.nodes()):
        labels[node] = idx    


    fig_size = plt.rcParams["figure.figsize"]  
    fig_size[0] = 15
    fig_size[1] = 6
    plt.rcParams["figure.figsize"] = fig_size  

    ax1= subplot(1,2,1)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos, labels, font_size=14)

    ax2 = subplot(1,2,2)
    im=ax2.imshow(GNet)
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04) 

    plt.show()


#INPUT NODES

def input_nodes(N_res, N_inp, rand = False):
    '''
    Returns a weight matrix for randomized input nodes. 
    
    Parameters:
    N_res: (int) Number of nodes in reservoir
    N_inp: (int) Number of input nodes
    rand: (bool) If input weights should have randomized weight
    '''
    InpNodeList = random.sample(range(N_res), N_inp) #select random nodes to be input nodes
    W_inp = np.zeros(N_res)
    
    if rand: #Input Node weights being randomized
        W_inp[InpNodeList] = np.random.random(N_inp)
    else: #Input Node weight full
        W_inp[InpNodeList] = np.ones(N_inp)
    
    return W_inp


#RESERVOIR

def Reservoir(GNet, U, W_inp, N_res, alpha, Init = 'false'):
    '''
    Calculates a time step in the Reservoir
    
    Parameters:
    GNet: (Numpy Array) Reservoir Matrix
    U: (Numpy array) Input Signal
    W_inp: (numpy array) Weights of the nodes
    N_res: (int) number of nodes in reservoir
    alpha: (float) Memory influence, alpha between 0 and 1
    '''
    N_U = len(U) #Number of points in the sequence

    R = np.zeros([N_res, N_U+1]) #Reservoir Computed Signal Sequence - at start empty
    #Inital state of the reservoir
    if type(Init) is np.ndarray:
        R[:,0] = Init
    
    #### time loop
    for t in range(0, N_U):      #Calculates for all points the equivalent mapped point
        R[:,t+1] = (1 - alpha)*np.asarray(R[:,t]) + alpha*np.tanh(np.dot(GNet, R[:,t].T) + W_inp*U[t] ) # (1-alpha)*old_state_reservoir + alpha*(new_state+new_input)

    return R


def reservoir_output(GNet, W_inp, N, alpha, X, Init='false'):
    '''
    Returns the output signal of the reservoir for an input sequence

    Parameters:
    GNet: (Numpy Array) Reservoir Matrix
    W_inp: (numpy array) Weights of the nodes
    N_res: (int) number of nodes in reservoir
    alpha: (float) Memory influence, alpha between 0 and 1
    X: (array) Array containing the training sequences
    '''
    X_out = []
    # Obtain output of the Reservoir for every sequence
    for sequence in X:
        R_out = Reservoir(GNet, sequence, W_inp, N, alpha, Init)
        X_out.append(R_out[:, -1])  # Append the last signal to the new matrix

    return X_out