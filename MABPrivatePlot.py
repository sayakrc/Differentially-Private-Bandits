# plot function (code taken from https://github.com/vaswanis/randucb)


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import os



mpl.rcParams["axes.linewidth"] = 0.75
mpl.rcParams["grid.linewidth"] = 0.75
mpl.rcParams["lines.linewidth"] = 0.75
mpl.rcParams["patch.linewidth"] = 0.75
mpl.rcParams["xtick.major.size"] = 3
mpl.rcParams["ytick.major.size"] = 3

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.size"] = 14
mpl.rcParams["axes.titlesize"] = "large"
mpl.rcParams["legend.fontsize"] = "large"

mpl.rcParams["text.usetex"] = True
mpl.rcParams['text.latex.preamble'] = [
    r'\usepackage{amsmath}',
    r'\usepackage{amssymb}']

print("matplotlib %s" % mpl.__version__)


def linestyle2dashes(style):
    if style == 'solid':
        return (10, ())
    elif style == 'dotted':
        return (0, (1, 1))
    elif style == 'loosely dotted':
        return (0, (1, 10))
    elif style == 'densely dotted':
        return (0, (1, 1))
    elif style == 'dashed':
        return (0, (5, 5))
    elif style == 'loosely dashed':
        return (0, (5, 10))
    elif style == 'densely dashed':
        return (0, (5, 1))
    elif style == 'dashdotted':
        return (0, (3, 5, 1, 5))
    elif style == 'loosely dashdotted':
        return (0, (3, 10, 1, 10))
    elif style == 'densely dashdotted':
        return (0, (3, 1, 1, 1))
    elif style == 'dashdotdotted':
        return (0, (3, 5, 1, 5, 1, 5))
    elif style == 'loosely dashdotdotted':
        return (0, (3, 10, 1, 10, 1, 10))
    elif style == 'densely dashdotdotted':
        return (0, (3, 1, 1, 1, 1, 1))
    

n = 10**6 # change to 10**7 for hard instance

gap = 1000 # sparsity
step = np.arange(1, n + 1, gap)
sube = (step.size // 10) * np.arange(1, 11) - 1

environments = [

     # easy instance (run it for Figure 1,2)
     ("Gaussian (easy, K = 10, eps = 0.1)"),
     ("Gaussian (easy, K = 10, eps = 0.5)"),
     ("Gaussian (easy, K = 10, eps = 1)"),
    
    # hard instance (run it for Figure 3)
     ("Gaussian (hard, K = 10, eps = 0.1)"),
     ("Gaussian (hard, K = 10, eps = 0.5)"),
     ("Gaussian (hard, K = 10, eps = 1)"),

    # SE algorithms (run it for Figure 4)
     ("Gaussian (easy-SE, K = 10, eps = 0.1)"),
     ("Gaussian (easy-SE, K = 10, eps = 0.5)"),
     ("Gaussian (easy-SE, K = 10, eps = 1)"),
]

algorithms = [
     
    # Algorithms to run for Figure 1,2,3
    ("DPSE(dc=2)", "black", "solid", "DP-SE"),
    ("BatchSEDistDP(dc=4)", "blue", "solid", "Dist-DP-SE"),
    ("BatchSEDistRDP(s=10,dc=4)", "red", "solid", "Dist-RDP-SE(s=10)"),
    ("BatchSEDistRDP(s=100,dc=4)", "green", "solid", "Dist-RDP-SE(s=100)"),
    
    # Algorithms to run for Figure 4
    ("BatchSECDP(dc=2)", "black", "solid", "CDP-SE"),
    ("BatchSELDP(dc=2)", "cyan", "solid", "LDP-SE"),
    ("BatchSEDistDP(dc=2)", "blue", "solid", "Dist-DP-SE"),
    ("BatchSEDistRDP(s=10,dc=2)", "red", "solid", "Dist-RDP-SE(s=10)"),
    ("BatchSEDistCDP(s=10,dc=2)", "green", "solid", "Dist-CDP-SE(s=10)"),

]


for env_def in environments:
    env_name = env_def
    res_dir = os.path.join("PathtoFile", "Results", "MAB", env_name)
    plt.figure(figsize=(5, 3))
    plt.subplot(1, 1, 1)
    for alg_idx, alg_def in enumerate(algorithms):
        alg_name, alg_color, alg_line, alg_label = alg_def[0], alg_def[1], alg_def[2], alg_def[3]
        
        fname = os.path.join(res_dir, alg_name)
        cum_regret = np.loadtxt(fname, delimiter=",")
        std_regret = cum_regret.std(axis=1) / np.sqrt(cum_regret.shape[1])

        lower =  cum_regret.mean(axis=1) - std_regret
        upper =  cum_regret.mean(axis=1) + std_regret

        lower_sparse = [lower[i] for i in step]
        upper_sparse = [upper[i] for i in step]
        cum_regret_sparse = [cum_regret.mean(axis=1)[i] for i in step]

        plt.fill_between(step,
                         lower_sparse,
                         upper_sparse,
                         color=alg_color, alpha=0.2, linewidth=0, rasterized=True)
        
        plt.plot(step, cum_regret_sparse, alg_color, linestyle=linestyle2dashes(alg_line), 
                 label=alg_label)
        
        
    #plt.ylim([0, 0.045])    #uncomment this line for hard instance
    
    plt.xlabel(r"Round")
    plt.ylabel("Time-average Regret") 
    plt.legend(loc= 'best', fontsize=12)
    plt.grid(True)
          
        
    plot_dir = os.path.join(".", "Plots", "MAB", env_name)
    os.makedirs(plot_dir, exist_ok=True)

    fig_name = "benchmarking.pdf"
    fname = os.path.join(plot_dir, fig_name)
    plt.savefig(fname, format = "pdf", dpi = 1200, bbox_inches="tight")
    plt.show()    