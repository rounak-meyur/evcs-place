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
                  show = True, fontsize=40, 
                  )
for conn in ["nearest", "optimal"]:
    synth_net, evcs = fx.read_inputs()
    initial = sum([synth_net.edges[e]["length"] \
                        for e in synth_net.edges])
    # additional edges for routing
    synth_net,flag = fx.connect_evcs(
        synth_net, evcs, 
        connection=conn,
        lambda_ = 1e2, 
        epsilon=1e-1,
        solver="gurobi", verbose=True)
    final = sum([synth_net.edges[e]["length"] \
                        for e in synth_net.edges])
    add = final - initial
    fx.plot_synth_net(synth_net, 
                      suptitle_sfx = f"{conn.title()} routing of EVCS : additional length = {add : 0.2f} meters",
                      to_file = f"{fx.area}_evcs_{conn}_connection", 
                      show = True, fontsize=40,
                      )