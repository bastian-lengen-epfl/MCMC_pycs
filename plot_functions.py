import numpy as np
import matplotlib.pyplot as plt

def plot_chain(chain, param_list):
    X2_list, pos_list, vel_list = chain

    f, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=False, sharey=False)
    ax = axes[0]
    ax.plot(np.log10(-np.array(X2_list)))
    ax.set_title('-logL')

    ax = axes[1]
    pos = np.array(pos_list)
    vel = np.array(vel_list)
    n_iter = len(pos)
    plt.figure()
    for i in range(0,len(pos[0])):
        ax.plot((pos[:,i]-pos[n_iter-1,i]),label=param_list[i])
    ax.set_title('particle position')
    ax.legend()

    ax = axes[2]
    for i in range(0,len(vel[0])):
        ax.plot(vel[:,i], label=param_list[i])
    ax.set_title('param velocity')
    ax.legend()
    return f, axes
