import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

m2km = 1e-3

# Use LaTeX-like font
rc('font', family='serif', serif=['Computer Modern Roman'])
rc('text', usetex=True)
fontsize = 18
fontsize_legend = 15


def plot_statistics(SAC_dict, OCP_dict):
    # Get initial dict (testdata)
    init_dict = SAC_dict['initial_dict']

    # Get trajectories
    X0_list = init_dict['X']
    Xsac_list = SAC_dict['optim_dict']['X']
    results_OCP = OCP_dict['results_dict']

    # Number of test
    N = len(Xsac_list)

    # Masks for the three cases
    collision = init_dict['collision'][:N]
    escape    = init_dict['escape'][:N]

    # Obtain masks
    mask_safe  = [not c and not e for c, e in zip(collision, escape)]
    mask_collision = [c and not e for c, e in zip(collision, escape)]
    mask_escape    = [e and not c for c, e in zip(collision, escape)]

    masks  = [mask_safe, mask_collision, mask_escape]
    titles = ["Natural stable",
              "Natural collision",
              "Natural escape"]

    # Natural trajectories collisions/escapes
    natural_collision = sum(init_dict['collision'][:N])
    natural_escape = sum(init_dict['escape'][:N])
    natural_safe = N - natural_collision - natural_escape

    # SAC
    # Non-success mask (True if solver did NOT succeed)
    statuses = SAC_dict['optim_dict']['status']
    non_success = [s != "success" for s in statuses]

    # Non-success among collisions / escapes
    sac_collision = sum(ns and c for ns, c in zip(non_success, init_dict['collision'][:N]))
    sac_escape = sum(ns and e for ns, e in zip(non_success, init_dict['escape'][:N]))
    sac_safe = N - sac_escape - sac_collision

    # Direct
    # Non-success mask (True if solver did NOT succeed)
    statuses = [d['status'] for d in OCP_dict['results_dict']]
    non_success = [s != "success" for s in statuses]

    # Non-success among collisions / escapes
    direct_collision = sum(ns and c for ns, c in zip(non_success, init_dict['collision'][:N]))
    direct_escape = sum(ns and e for ns, e in zip(non_success, init_dict['escape'][:N]))
    direct_safe = N - direct_escape - direct_collision

    # Bar positions
    categories = ['Stable', 'Collision', 'Escape']
    x = np.arange(len(categories))
    width = 0.3

    fig, ax = plt.subplots(figsize=(6, 5))

    # Plot SAC bars (shift left)
    ax.bar(x - width,
           [sac_safe/N * 100, sac_collision/N * 100, sac_escape/N * 100],
           width, label='SAC', color='blue')

    # Plot Direct bars (centered)
    ax.bar(x,
           [direct_safe/N * 100, direct_collision/N * 100, direct_escape/N * 100],
           width, label='OCP', color='green')

    # Plot Natural bars (shift right)
    ax.bar(x + width,
           [natural_safe/N * 100, natural_collision/N * 100, natural_escape/N * 100],
           width, label='Natural', color='orange')

    # Labels
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=fontsize)
    ax.set_yticks(range(0, 101, 10))
    ax.set_ylabel('Percentage [\%]', fontsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.legend(fontsize=fontsize_legend)
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.savefig("images/percentage.eps", bbox_inches="tight")
