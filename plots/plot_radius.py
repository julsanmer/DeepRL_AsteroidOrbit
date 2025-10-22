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


def plot_radius(SAC_dict, OCP_dict, N):
    # Natural
    Tnat_list = SAC_dict['initial_dict']['T']
    Xnat_list = SAC_dict['initial_dict']['X']

    # SAC
    Tsac_list = SAC_dict['optim_dict']['T'][0:N]
    Xsac_list = SAC_dict['optim_dict']['X'][0:N]

    fig, (ax_sac, ax_direct) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for i in range(N):
        # Positions
        t_nat = Tnat_list[i]
        t_sac = Tsac_list[i]
        t_direct = OCP_dict['results_dict'][i]['Tsim']
        pos_nat = Xnat_list[i]
        pos_sac = Xsac_list[i]
        pos_direct = OCP_dict['results_dict'][i]['Xsim'][:, 0:3]

        # Radius
        r_nat = np.linalg.norm(pos_nat, axis=1)
        r_sac = np.linalg.norm(pos_sac, axis=1)
        r_direct = np.linalg.norm(pos_direct, axis=1)

        # Plot SAC subplot
        ax_sac.plot(t_nat * s2h, r_nat * m2km,
                    color='orange', zorder=10, alpha=0.5)
        ax_sac.plot(t_sac * s2h, r_sac * m2km,
                    color='blue', alpha=1)

        # Plot Direct subplot
        ax_direct.plot(t_nat * s2h, r_nat * m2km,
                       color='orange', zorder=10, alpha=0.5)
        ax_direct.plot(t_direct * s2h, r_direct * m2km,
                       color='green', alpha=1)
        ax_sac.axhline(y=22, color='red', linestyle='--',
                       linewidth=2, zorder=100)
        ax_sac.axhline(y=30, color='red', linestyle='--',
                       linewidth=2, zorder=100)
        ax_direct.axhline(y=22, color='red', linestyle='--',
                          linewidth=2, zorder=100)
        ax_direct.axhline(y=30, color='red', linestyle='--',
                          linewidth=2, zorder=100)

    # SAC subplot formatting
    ax_sac.set_title("SAC", fontsize=fontsize)
    ax_sac.tick_params(axis='x', labelsize=fontsize)
    ax_sac.tick_params(axis='y', labelsize=fontsize)
    ax_sac.set_xlim(t_sac[0] * s2h, t_sac[-1] * s2h)
    ax_sac.set_ylim(10, 50)
    ax_sac.grid(True)

    # Direct subplot formatting
    ax_direct.set_title("OCP", fontsize=fontsize)
    ax_direct.tick_params(axis='x', labelsize=fontsize)
    ax_direct.tick_params(axis='y', labelleft=False)  # hide y-axis labels
    ax_direct.set_xlim(t_sac[0] * s2h, t_sac[-1] * s2h)
    ax_direct.set_ylim(10, 50)
    ax_direct.grid(True)

    # Central labels
    fig.text(0.53, 0.04, "Time [h]", ha='center',
             fontsize=fontsize)
    fig.text(0.03, 0.5, "Orbital radius [km]", va='center',
             rotation='vertical', fontsize=fontsize)

    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])  # leave space for central labels and legend
    plt.savefig("images/radius.pdf", bbox_inches="tight")