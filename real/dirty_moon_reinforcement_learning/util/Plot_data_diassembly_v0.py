#!/usr/bin/env python3
'''
    By Xiazhen Xu <xxx212@student.bham.ac.uk>
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import os

script_dir = os.path.dirname(__file__)
results_dir = os.path.join(script_dir, "results/")
sample_file_name = "Disassembly_results"
sample_file_name_2 = "Force_realtime"


# def plot_learning_curve(x, scores):
#     plt.xlabel("Steps")
#     plt.ylabel("Reward")
#     plt.plot(x, scores)

#     plt.title('Reward plot')
#     plt.savefig(results_dir + sample_file_name)

def plot_learning_curve(x, scores):
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.plot(x, scores)

    plt.title('Reward plot')
    plt.savefig(results_dir + sample_file_name)

def plot_force_curve(x, force):
    plt.xlabel("Steps")
    plt.ylabel("Force")
    plt.plot(x, force)

    plt.title('Force plot')
    plt.savefig(results_dir + sample_file_name_2)



