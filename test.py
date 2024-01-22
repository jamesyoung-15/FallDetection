from utils import pose_utils
from itertools import combinations
import numpy as np

num_frames = 3
hips = np.array([(395, 175), (412, 223), (407, 222)])
spine = np.array([(-32, -81), (20, -80), (80, -14)])
# O(n^2) time complexity
for i in range(num_frames-1, -1, -1):
    for j in range(i-1, -1, -1):
        print(i,j)
        print(pose_utils.angle_between(spine[i], spine[j]))
        print(hips[i][1] - hips[j][1])
        print()        
