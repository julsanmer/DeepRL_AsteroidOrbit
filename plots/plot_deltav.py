import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

m2km = 1e-3
s2h = 1/3600

# Use LaTeX-like font
rc('font', family='serif', serif=['Computer Modern Roman'])
rc('text', usetex=True)
fontsize = 18
fontsize_legend = 15


def plot_deltav(SAC_dict, OCP_dict, N):
    # Natural
    Tnat_list = SAC_dict['initial_dict']['T']
    Xnat_list = SAC_dict['initial_dict']['X']

    # SAC
    Tsac_list = SAC_dict['optim_dict']['T'][0:N]
    Xsac_list = SAC_dict['optim_dict']['X'][0:N]
    dVsac_list = SAC_dict['optim_dict']['dV'][0:N]

    fig2, (ax_sac, ax_direct) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    t_sac = Tsac_list[0]
    for i in range(N):
        # Get dVs
        dV_sac = dVsac_list[i]
        dV_direct = OCP_dict['results_dict'][i]['dV']

        # Times
        Tsac_dV = np.linspace(t_sac[0], t_sac[-1], len(dV_sac))
        Tdirect_dV = np.linspace(t_sac[0], t_sac[-1], len(dV_direct))

        # L1 norm along axis=1
        l1_sac = np.linalg.norm(dV_sac, ord=1, axis=1)
        l1_direct = np.linalg.norm(dV_direct, ord=1, axis=1)

        # Cumulative delta-v
        cumdV_sac = np.cumsum(l1_sac)
        cumdV_direct = np.cumsum(l1_direct)

        # Plots
        ax_sac.plot(Tsac_dV * s2h, cumdV_sac, drawstyle='steps-post',
                    color='blue')
        ax_direct.plot(Tdirect_dV * s2h, cumdV_direct, drawstyle='steps-post',
                       color='green')

    # SAC subplot formatting
    ax_sac.set_title("SAC", fontsize=fontsize)
    ax_sac.tick_params(axis='x', labelsize=fontsize)
    ax_sac.tick_params(axis='y', labelsize=fontsize)
    ax_sac.set_xlim(t_sac[0] * s2h, t_sac[-1] * s2h)
    ax_sac.set_ylim(-0.2, 4)
    ax_sac.grid(True, linestyle='--', alpha=0.7)

    # Direct subplot formatting
    ax_direct.set_title("OCP", fontsize=fontsize)
    ax_direct.tick_params(axis='x', labelsize=fontsize)
    ax_direct.tick_params(axis='y', labelleft=False)  # hide y-axis labels
    ax_direct.set_xlim(t_sac[0] * s2h, t_sac[-1] * s2h)
    ax_direct.set_ylim(-0.2, 4)
    ax_direct.grid(True, linestyle='--')

    # Central labels
    fig2.text(0.53, 0.04, "Time [h]", ha='center', fontsize=fontsize)
    fig2.text(0.03, 0.5, "Cumulative $\Delta v$ [m/s]", va='center',
              rotation='vertical', fontsize=fontsize)

    plt.tight_layout(rect=[0.05, 0.05, 1, 1])
    plt.savefig("images/deltav.eps", bbox_inches="tight")