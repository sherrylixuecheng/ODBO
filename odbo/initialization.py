import numpy as np
#Not done yet


def initial_design(X_pending, least_occurance=2, target_num=None, random_state=0):
   np.randomseed(random_state) 
   return X_choice

def top_exp(X, Y, target_num=40):
   ranks = np.argsort(Y)[-target:]
   return X[ranks, :], Y[ranks]

def random_exp(X, Y, target_num=None, random_state= 0):
   np.randomseed(random_state)
   return X, Y

