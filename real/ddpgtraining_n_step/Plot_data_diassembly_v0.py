import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt


def plot_learning_curve(x, scores, figure_file):
    plt.xlabel("Steps")
    plt.ylabel("Reward")
    plt.plot(x, scores)

    plt.title('Reward plot')
    plt.savefig(figure_file)

