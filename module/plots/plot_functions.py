import numpy as np
import matplotlib.pyplot as plt
import corner

def plot_chain_PSO(chain, param_list):
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

def plot_chain_MCMC(theta, chi2, param_list):
    fig1 = corner.corner(theta, labels=param_list)

    fig2 = plt.figure(2)
    x = np.arange(len(chi2))
    plt.xlabel('N', fontdict={"fontsize": 16})
    plt.ylabel('$\chi^2$', fontdict={"fontsize": 16})
    plt.plot(x, chi2)

    fig3, axe = plt.subplots(2, 1, sharex=True)
    axe[0].plot(x, theta[:, 0], 'r')
    axe[1].plot(x, theta[:, 1], 'g')
    plt.xlabel('N', fontdict={"fontsize": 16})
    axe[0].set_ylabel(param_list[0], fontdict={"fontsize": 16})
    axe[1].set_ylabel(param_list[1], fontdict={"fontsize": 16})

    return fig1,fig2,fig3

def plot_chain_grid_dic(optimiser):
    x_param = optimiser.chain_list[0]
    fig1 = plt.figure(1)
    plt.errorbar(x_param, optimiser.chain_list[2][0], yerr=optimiser.chain_list[3][0])
    plt.hlines(optimiser.fit_vector[0], np.min(x_param), np.max(x_param), colors='r', linestyles='solid', label='target')
    plt.xlabel('B in unit of Nymquist frequency)')
    plt.ylabel('zruns')
    plt.legend()

    fig2 = plt.figure(2)
    plt.errorbar(x_param, optimiser.chain_list[2][1], yerr=optimiser.chain_list[3][1])
    plt.hlines(optimiser.fit_vector[1], optimiser.grid[0], optimiser.grid[-1], colors='r', linestyles='solid', label='target')
    plt.xlabel('B in unit of Nymquist frequency)')
    plt.ylabel('sigma')
    plt.legend()

    fig3 = plt.figure(3)
    plt.plot(x_param, optimiser.chain_list[1])
    plt.xlabel('B in unit of Nymquist frequency)')
    plt.ylabel('$\chi^2$')

    fig1.savefig(optimiser.savedirectory + optimiser.tweakml_name + '_zruns_' + optimiser.lc.object + '.png')
    fig2.savefig(optimiser.savedirectory + optimiser.tweakml_name + '_std_' + optimiser.lc.object + '.png')
    fig3.savefig(optimiser.savedirectory + optimiser.tweakml_name + '_chi2_' + optimiser.lc.object + '.png')

    if optimiser.display:
        plt.show()
    plt.clf()
    plt.close('all')