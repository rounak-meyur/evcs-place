
from pyEVCSlib import EVCSFixture


fx = EVCSFixture('runTest')
fx.out_dir = "out/script"
fx.fig_dir = "figs/script"
fx.grb_dir = "gurobi/script"
fx.area = 'Area 2'


fx.plot_investment(csv_file="demand_L1000000.csv", 
                    suptitle_sfx = "investment for routing power lines", 
                    to_file=f"{fx.area}-investment", 
                    show=False, fontsize=35)

fx.plot_improvement(csv_file="demand_L1000000.csv", 
                    suptitle_sfx = "improvement in voltages", 
                    to_file=f"{fx.area}-improvement", 
                    show=False, fontsize=35)