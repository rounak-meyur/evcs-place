from pyEVCSlib import EVCSFixture

fx = EVCSFixture('runTest')
fx.out_dir = "out/script"
fx.fig_dir = "figs/script"
fx.grb_dir = "gurobi/script"
fx.area = 'Area 6'


fx.demand = float(30 * 1000 / 24.0)
        
# initial read
synth_net, evcs = fx.read_inputs()

fx.plot_synth_net(synth_net, 
                  suptitle_sfx = "Synthetic distribution network",
                  to_file = f"{fx.area}_synthetic_net", 
                  show = True, fontsize=30
                  )

# additional edges for routing
synth_net,flag = fx.connect_evcs(
    synth_net, evcs, 
    connection="optimal",
    lambda_ = 1000, 
    epsilon=1e-1,
    solver="gurobi", verbose=True)

fx.plot_synth_net(synth_net, 
                  suptitle_sfx = "Optimal routing of EVCS",
                  to_file = f"{fx.area}_evcs_connection", 
                  show = True, fontsize=30,
                  )