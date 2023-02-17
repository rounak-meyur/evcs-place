from pyEVCSlib import EVCSFixture

fx = EVCSFixture('runTest')
fx.out_dir = "out/script"
fx.fig_dir = "figs/script"
fx.grb_dir = "gurobi/script"
fx.area = 'Area 2'


fx.demand = float(30 * 1000 / 24.0)
        
# initial read
synth_net, evcs = fx.read_inputs()

# additional edges for routing
synth_net = fx.connect_evcs(
    synth_net, evcs, 
    connection="optimal",
    lambda_ = 1e6, 
    epsilon=1e-1,
    solver="gurobi", verbose=True)
