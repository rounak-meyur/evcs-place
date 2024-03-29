import sys
from pyEVCSlib import EVCSFixture


fx = EVCSFixture('runTest')
fx.out_dir = "out/script"
fx.fig_dir = "figs/script"
fx.grb_dir = "gurobi/script"

arealist = [3, 6, 7, 9, 13, 14, 16, 17, 18, 19, 20]
arealist = [6]

adopt = [10, 40, 70, 100]

for area in arealist:
    fx.area = f'Area {area}'

    fx.plot_investment(csv_file=f"{fx.area}_summary.csv", 
                        suptitle_sfx = "Investment for routing power lines", 
                        to_file=f"{fx.area}-investment", 
                        show=False, fontsize=25, 
                        adoptions=adopt,
                        figsize=(15,8),
                        )

    fx.plot_improvement(csv_file=f"{fx.area}_summary.csv", 
                        suptitle_sfx = "Improvement in reliability", 
                        to_file=f"{fx.area}-improvement", 
                        show=False, fontsize=25, 
                        adoptions=adopt,
                        figsize=(15,8),
                        )

    fx.plot_tradeoff(csv_file=f"{fx.area}_summary.csv", 
                    suptitle_sfx = "Investment to reliability trade off", 
                    to_file=f"{fx.area}-tradeoff", 
                    show=False, fontsize=25, tick_fontsize=15, 
                    linewidth=5, markersize=20, 
                    adoptions=adopt, 
                    figsize=(25,8),
                    )