import numpy as np

# NORTH, EAST, SOUTH, WEST
E = np.array([0, 1])
N = np.array([1, 0])
W = -E
S = -N
NE = N + E
NW = N + W
SE = S + E
SW = S + W
