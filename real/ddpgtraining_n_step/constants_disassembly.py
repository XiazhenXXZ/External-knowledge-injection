import numpy as np
import math

# ====== COLORS ======#
Black = (0, 0, 0)
GRAY = (127, 127, 127)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
MAGENTA = (0, 255, 255)
CYAN = (255, 0, 255)

# ====== robot_parameter ======#
Gravity = -10

# ====== observation ======#
min_EEP_x = 0  # EEP = End Effector Position
min_EEP_y = 0
min_EEP_z = 0.1
max_EEP_x = 1
max_EEP_y = 0.05
max_EEP_z = math.inf

min_Ori_r = 1.3  # Ori = orientation
min_Ori_p = -1.2
min_Ori_y = -math.inf
max_Ori_r = 1.8
max_Ori_p = 1.2
max_Ori_y = math.inf

min_F_x = -math.inf
min_F_y = -math.inf
min_F_z = -math.inf
max_F_x = math.inf
max_F_y = math.inf
max_F_z = math.inf

min_T_x = -math.inf
min_T_y = -math.inf
min_T_z = -math.inf
max_T_x = math.inf
max_T_y = math.inf
max_T_z = math.inf

min_x = -math.inf
min_y = -math.inf
min_z = -math.inf
min_r = -math.inf
min_p = -math.inf
min_y_ = -math.inf
min_D1 = -math.inf
min_O1 = -math.inf

max_x = math.inf
max_y = math.inf
max_z = math.inf
max_r = math.inf
max_p = math.inf
max_y_ = math.inf
max_D1 = math.inf
max_O1 = math.inf