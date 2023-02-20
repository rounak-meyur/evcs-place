from pyEVCSlib import EVCSFixture
from pyUtilslib import powerflow
import pandas as pd


fx = EVCSFixture('runTest')
fx.out_dir = "out/script"
fx.fig_dir = "figs/script"
fx.grb_dir = "gurobi/script"


# area_list = []
# for i in range(20):
#     fx.area = f'Area {i+1}'
#     synth_net, evcs = fx.read_inputs()
#     if len(evcs.cord) > 0:
#         area_list.append(fx.area)


volt_range = [0.97, 0.95, 0.92, 0.90]
lambda_list = [1e-6, 1, 1e6]
rating_list = [30, 50, 100, 120, 150, 180, 250, 350]
area_list = ['Area 1', 'Area 3', 'Area 4', 'Area 6', 'Area 7', 'Area 8', 
             'Area 9', 'Area 10', 'Area 11', 'Area 12', 'Area 13', 'Area 14', 
             'Area 15', 'Area 16', 'Area 17', 'Area 18', 'Area 19', 'Area 20']
        



for area in area_list[11:]:
    fx.area = area

    data = {"lambda":[], "rating":[], "connection":[], "length":[]}
    data.update({f"< {v}":[] for v in volt_range})
    for conn_type in ["nearest", "optimal"]:
    
        for lambda_ in lambda_list:
            for k in range(len(rating_list)):
                
                print(f"Computing for {fx.area} : connection={conn_type} : lambda={lambda_} : rating={rating_list[k]}")
                fx.demand = float(rating_list[k] * 1000 / 24.0)
                
                data["lambda"].append(lambda_)
                data["connection"].append(conn_type)
                data["rating"].append(rating_list[k])
                
                # initial read
                synth_net, evcs = fx.read_inputs()
                init_length = sum([synth_net.edges[e]["length"] \
                                    for e in synth_net.edges])
                
                # additional edges for routing
                synth_net,opt_flag = fx.connect_evcs(
                    synth_net, evcs, 
                    connection=conn_type,
                    lambda_ = lambda_, 
                    epsilon=1e-1,)
                
                if opt_flag:
                    final_length = sum([synth_net.edges[e]["length"] \
                                        for e in synth_net.edges])
                    
                    # Evaluate the additional length
                    add_length = final_length - init_length
                
                    # Add it to the data
                    data["length"].append(add_length)
                
                    # run powerflow and number of nodes outside limit
                    powerflow(synth_net)
                    nodelist = [n for n in synth_net if synth_net.nodes[n]['label']!='R']
                    for v in volt_range:
                        num_nodes = len([n for n in nodelist if synth_net.nodes[n]["voltage"] < v])
                        data[f"< {v}"].append(num_nodes * 100.0 / len(nodelist))
                else:
                    data["length"].append(float("nan"))
                    for v in volt_range:
                        data[f"< {v}"].append(float("nan"))
    
    # Create the dataframe
    df = pd.DataFrame(data)
    df.to_csv(f"{fx.out_dir}/{fx.area}_summary.csv", index=False)