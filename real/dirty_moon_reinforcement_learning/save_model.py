#!/usr/bin/env python3
'''
    By Xiazhen Xu <xxx212@student.bham.ac.uk>
'''

import os
import gymnasium as gym
import time
import torch
import torch.optim as optim

from algorithm.dqn_pytorch import Agent, DeepQNetwork
from env.disassembly_env_v0 import DisassemblyEnv

model = DeepQNetwork
filepath = os.getcwd()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

if not os.path.exists('torch_directory'):
    os.makedirs('torch_directory')

def save_model(model_directory):
    checkpoint_directory = 'torch_directory'
    file_path = os.path.join(checkpoint_directory, 'model.pt')
    torch.save(model_directory, file_path)

model_dictionary = {'model_state': model.state_dict(),
                    'model_optimizer': optimizer.state_dict()}

save_model(model_dictionary)



