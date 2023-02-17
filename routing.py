from pyEVCSlib import EVCSFixture
from pyUtilslib import powerflow
import pandas as pd
import tqdm

fx = EVCSFixture('runTest')
fx.out_dir = "out/script"
fx.fig_dir = "figs/script"
fx.grb_dir = "gurobi/script"
fx.area = 'Area 2'


volt_range = [0.97, 0.95, 0.92, 0.90]
data = {"rating":[], "connection":[], "length":[]}
data.update(
    {f"< {v}":[] for v in volt_range}
    )
lambda_ = 1000000
rating_list = [30, 50, 100, 120, 150, 180, 250, 350]

for conn_type in ["optimal", "nearest"]:

    for k in tqdm(range(len(rating_list)), 
                  desc="Computing for different EV charger ratings",
                  ncols=100):
        fx.demand = float(rating_list[k] * 1000 / 24.0)
        
        # initial read
        synth_net, evcs = fx.read_inputs()
        init_length = sum([synth_net.edges[e]["length"] \
                            for e in synth_net.edges])
        
        # additional edges for routing
        synth_net = fx.connect_evcs(
            synth_net, evcs, 
            connection=conn_type,
            lambda_ = lambda_, 
            epsilon=1e-1,)
        final_length = sum([synth_net.edges[e]["length"] \
                            for e in synth_net.edges])
        
        # Evaluate the additional length
        add_length = final_length - init_length
        
        # Add it to the data
        data["connection"].append(conn_type)
        data["rating"].append(rating_list[k])
        data["length"].append(add_length)
        
        # run powerflow and number of nodes outside limit
        powerflow(synth_net)
        nodelist = [n for n in synth_net if synth_net.nodes[n]['label']!='R']
        for v in volt_range:
            num_nodes = len([n for n in nodelist if synth_net.nodes[n]["voltage"] < v])
            data[f"< {v}"].append(num_nodes * 100.0 / len(nodelist))

# Create the dataframe
df = pd.DataFrame(data)
df.to_csv(f"{fx.out_dir}/{fx.area}_L{lambda_}_summary.csv", index=False)