import numpy as np
import matplotlib.pyplot as plt
import corner

def plot_chain_PSO(chain, param_list):
    X2_list, pos_list, vel_list = chain

    f, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=False, sharey=False)
    ax = axes[0]
    ax.plot(np.log10(np.array(X2_list)))
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

def corner_plot_MCMC(theta, param_list):
    fig1 = corner.corner(theta, labels=param_list)
    return fig1


def plot_chain_MCMC(theta, chi2, param_list):
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

    return fig2,fig3

def plot_chain_grid_dic(optimiser):
    for i,l in enumerate(optimiser.lcs) :
        x_param = np.asarray(optimiser.explored_param)[:,i]
        z_runs = np.asarray( optimiser.chain_list[2])[:,i]
        z_runs_err = np.asarray(optimiser.chain_list[4])[:,i]
        sigmas = np.asarray(optimiser.chain_list[3])[:,i]
        sigmas_err = np.asarray(optimiser.chain_list[5])[:,i]

        fig1 = plt.figure(1)
        plt.errorbar(x_param[:,0], z_runs, yerr=z_runs_err,marker ='o')
        plt.hlines(optimiser.fit_vector[i,0], np.min(x_param[:,0]), np.max(x_param[:,0]), colors='r', linestyles='solid', label='target')
        plt.xlabel('B in unit of Nymquist frequency)')
        plt.ylabel('zruns')
        plt.legend()

        fig2 = plt.figure(2)
        plt.errorbar(x_param[:,0], sigmas, yerr=sigmas_err, marker ='o')
        plt.hlines(optimiser.fit_vector[i,1], np.min(x_param[:,0]), np.max(x_param[:,0]), colors='r', linestyles='solid', label='target')
        plt.xlabel('B in unit of Nymquist frequency)')
        plt.ylabel('sigma')
        plt.legend()

        fig1.savefig(optimiser.savedirectory + optimiser.tweakml_name + '_zruns_' + l.object + '.png')
        fig2.savefig(optimiser.savedirectory + optimiser.tweakml_name + '_std_' + l.object + '.png')

        if optimiser.display:
            plt.show()
        plt.clf()
        plt.close('all')

    fig3 = plt.figure(3)
    x = np.arange(1,len(optimiser.chain_list[1])+1,1)
    plt.plot(x, optimiser.chain_list[1])
    plt.xlabel('Iteration')
    plt.ylabel('$\chi^2$')
    fig3.savefig(optimiser.savedirectory + optimiser.tweakml_name + '_chi2.png')
