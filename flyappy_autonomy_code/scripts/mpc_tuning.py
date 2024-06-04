import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    Xk_pd = pd.read_csv("Xk.csv")
    Uk_pd = pd.read_csv("Uk.csv")

    Xk = np.array(Xk_pd.values)
    Uk = np.array(Uk_pd.values)

    fig, ax = plt.subplots(Xk.shape[1] + Uk.shape[1])
    ax[0].plot(Xk[:, 0])
    ax[0].set_title("x")
    ax[1].plot(Xk[:, 1])
    ax[1].set_title("x vel")
    ax[2].plot(Xk[:, 2])
    ax[2].set_title("y")
    ax[3].plot(Xk[:, 3])
    ax[3].set_title("y vel")
    ax[4].plot(Uk[:, 0])
    ax[4].set_title("ax")
    ax[5].plot(Uk[:, 1])
    ax[5].set_title("ay")
    plt.show()
    pass