import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

from .dv_rtn import cartesian_to_rtn


m2km = 1e-3


# Use LaTeX-like font
rc('font', family='serif', serif=['Computer Modern Roman'])
rc('text', usetex=True)
fontsize = 18
fontsize_legend = 15


def plot_dvrtn(SAC_dict, OCP_dict):
    # Get initial dict (testdata)
    init_dict = SAC_dict['initial_dict']

    # Get trajectories
    Xsac_list = SAC_dict['optim_dict']['X']
    dVsac_list = SAC_dict['optim_dict']['dV']
    results_direct = OCP_dict['results_dict']

    # Number of test
    N = len(Xsac_list)

    # Masks for the three cases
    collision = init_dict['collision'][:N]
    escape = init_dict['escape'][:N]

    # Obtain masks
    mask_safe = [not c and not e for c, e in zip(collision, escape)]
    mask_collision = [c and not e for c, e in zip(collision, escape)]
    mask_escape = [e and not c for c, e in zip(collision, escape)]

    masks = [mask_safe, mask_collision, mask_escape]
    titles = ["Natural stable",
              "Natural collision",
              "Natural escape"]
    components = [r'Radial', r'Tangential', r'Normal']

    # Asteroid angular velocity
    omega = OCP_dict['initial_dict']['asteroid']['omega']

    fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True)
    for i, (mask, title) in enumerate(zip(masks, titles)):
        try:
            # Example data
            arr = np.arange(0, 36001, 6)  # [0, 36, 72, ..., 36000]

            # Mask for values that are multiples of 600 (including 0)
            mask_time = (arr % 600 == 0)
            idx = mask.index(True)

            # Compute RTN dv for SAC
            pos_sac = Xsac_list[idx][mask_time, :3]
            vel_sac = Xsac_list[idx][mask_time, 3:]
            dv_sac = dVsac_list[idx]
            dvrtn_sac = cartesian_to_rtn(dv_sac, pos_sac[:-1],
                                         vel_sac[:-1], omega)

            # Compute RTN dv for OCP
            pos_ocp = results_direct[idx]['X'][:, :3]
            vel_ocp = results_direct[idx]['X'][:, 3:]
            dv_ocp = results_direct[idx]['dV'][:]
            dvrtn_ocp = cartesian_to_rtn(dv_ocp, pos_ocp[:-1],
                                         vel_ocp[:-1], omega)

            # Sum of absolute values per column
            dvsum_sac = np.abs(dvrtn_sac).sum(axis=0)
            dvsum_ocp = np.abs(dvrtn_ocp).sum(axis=0)

            # Labels
            width = 0.35
            x = np.arange(len(components))
            axes[i].bar(x - width/2, np.abs(dvsum_sac), width, label='SAC', color='blue')
            axes[i].bar(x + width/2, np.abs(dvsum_ocp), width, label='OCP', color='green')
            axes[i].set_xticks(x)
            axes[i].set_xticklabels(components)
            if i == 0:
                axes[i].set_ylabel(r'Cumulative delta-$v$ [m/s]', fontsize=fontsize)
                axes[i].legend(fontsize=fontsize_legend)
            axes[i].tick_params(axis='both', labelsize=fontsize)
            axes[i].set_title(titles[i], fontsize=fontsize)
            axes[i].grid(True, linestyle='--', alpha=0.5)
        except ValueError:
            axes[i].text(0.5, 0.5, 0.5, "No case", transform=axes[i].transAxes)
    plt.tight_layout()
    plt.savefig("images/dvrtn.pdf", bbox_inches="tight", pad_inches=0.4)
