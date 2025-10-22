import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

m2km = 1e-3

# Use LaTeX-like font
rc('font', family='serif', serif=['Computer Modern Roman'])
rc('text', usetex=True)
fontsize = 18
fontsize_legend = 15


def plot_orbits(SAC_dict, OCP_dict, shape):
    # Get initial dict (testdata)
    init_dict = SAC_dict['initial_dict']

    # Get trajectories
    X0_list = init_dict['X']
    Xsac_list = SAC_dict['optim_dict']['X']
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

    fig, axes = plt.subplots(1, 3,
                             subplot_kw={'projection': '3d'}, figsize=(16, 6))

    # To store handles and labels for shared legend
    shared_handles = []
    shared_labels = []

    for i, (ax, mask, title) in enumerate(zip(axes, masks, titles)):
        try:
            idx = mask.index(True)
            pos0 = X0_list[idx][:, :3]
            pos_sac = Xsac_list[idx][:, :3]
            pos_ocp = results_direct[idx]['Xsim'][:, :3]

            h1, = ax.plot(pos_sac[:, 0] * m2km, pos_sac[:, 1] * m2km, pos_sac[:, 2] * m2km,
                          label="SAC", color='blue')
            h2, = ax.plot(pos_ocp[:, 0] * m2km, pos_ocp[:, 1] * m2km, pos_ocp[:, 2] * m2km,
                          label="OCP", color='green', linestyle='--')
            h3, = ax.plot(pos0[:, 0] * m2km, pos0[:, 1] * m2km, pos0[:, 2] * m2km,
                          label="Natural", color='orange')

            # Add square markers at the last point
            ax.plot(pos_sac[-1, 0] * m2km, pos_sac[-1, 1] * m2km, pos_sac[-1, 2] * m2km,
                    marker='o', color='blue', linestyle='')
            ax.plot(pos_ocp[-1, 0] * m2km, pos_ocp[-1, 1] * m2km, pos_ocp[-1, 2] * m2km,
                    marker='o', color='green', linestyle='--')
            ax.plot(pos0[-1, 0] * m2km, pos0[-1, 1] * m2km, pos0[-1, 2] * m2km,
                    marker='o', color='orange', linestyle='')
            ax.view_init(elev=40, azim=-50)
            shape.plot3D(ax, scale=m2km, color="#4f4f4f")

            # store handles/labels from first subplot only
            if not shared_handles:
                shared_handles.extend([h1, h2, h3])
                shared_labels.extend([h1.get_label(),
                                      h2.get_label(),
                                      h3.get_label()])
                ax.legend(shared_handles, shared_labels, loc='upper right',
                          fontsize=fontsize_legend)

        except ValueError:
            ax.text(0.5, 0.5, 0.5, "No case", transform=ax.transAxes)
        ax.set_title(title, fontsize=fontsize)
        ax.set_xlabel('$x$ [km]', fontsize=fontsize)
        ax.set_ylabel('$y$ [km]', fontsize=fontsize)
        ax.set_zlabel('$z$ [km]', fontsize=fontsize)
        ax.tick_params(axis='x', labelsize=fontsize)
        ax.tick_params(axis='y', labelsize=fontsize)
        ax.tick_params(axis='z', labelsize=fontsize)
        ax.set_xlim(-30, 30)
        ax.set_ylim(-30, 30)
        ax.set_zlim(-30, 30)
    plt.tight_layout(pad=4.0)
    plt.savefig("images/3Dorbits.pdf", bbox_inches="tight", pad_inches=0.4)
