import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import subprocess as sp
import math

def main():
    X = np.arange(256, 2049, 256)
    Y = np.arange(256, 2049, 256)
    X, Y = np.meshgrid(X, Y)
    Z = X.copy()
    csv = pd.read_csv("test2_64.csv", sep=";")
    n = int(math.sqrt(len(csv)))
    for i in range(n):
        for j in range(n):
            Z[i][j] = csv["time"][i*n + j]
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.viridis)
    plt.show()

if __name__ == '__main__':
    main()
