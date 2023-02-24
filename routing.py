import os, sys
from pyEVCSlib import EVCSFixture
from pyUtilslib import powerflow
import pandas as pd


fx = EVCSFixture('runTest')
fx.out_dir = "out/script"
fx.fig_dir = "figs/script"
fx.grb_dir = "gurobi/script"

def update_data(data_dict, net, flag, v_range, length):
    if flag:
        final_length = sum([net.edges[e]["length"] \
                            for e in net.edges])
        
        # Evaluate the additional length
        add_length = final_length - length
    
        # Add it to the data
        data_dict["length"].append(add_length)
    
        # run powerflow and number of nodes outside limit
        powerflow(net)
        nodelist = [n for n in net if net.nodes[n]['label']!='R']
        for v in v_range:
            num_nodes = len([n for n in nodelist if net.nodes[n]["voltage"] < v])
            data_dict[f"< {v}"].append(num_nodes * 100.0 / len(nodelist))
    else:
        data_dict["length"].append(float("nan"))
        for v in v_range:
            data_dict[f"< {v}"].append(float("nan"))
    return data_dict


# area_list = []
# for i in range(20):
#     fx.area = f'Area {i+1}'
#     synth_net, evcs = fx.read_inputs()
#     if len(evcs.cord) > 0:
#         area_list.append(fx.area)


volt_range = [0.97, 0.95, 0.92, 0.90]
lambda_list = [1e-6, 1, 1e6]
rating_list = [30, 50, 100, 120, 150, 180, 250, 350]
conn_list = ["nearest", "optimal"]
# area_list = ['Area 1', 'Area 3', 'Area 4', 'Area 6', 'Area 7', 'Area 8', 
#              'Area 9', 'Area 10', 'Area 11', 'Area 12', 'Area 13', 'Area 14', 
#              'Area 15', 'Area 16', 'Area 17', 'Area 18', 'Area 19', 'Area 20']
        
# area_list = ['Area 1', 'Area 4', 'Area 8', 'Area 11', 'Area 12', 'Area 15']
area_list = ['Area 6']
lambda_list = [10, 100, 1000, 10000, 100000]
rating_list = [30, 50, 100, 120, 150, 180, 250, 350]
conn_list = ["optimal"]

for area in area_list:
    fx.area = area
    # check if csv file for the area exists
    filename = f"{fx.out_dir}/{fx.area}_summary.csv"
    if os.path.exists(filename):
        df = pd.read_csv(filename, header=0)
        data = df.to_dict(orient='list')
    else:
        data = {"lambda":[], "rating":[], "connection":[], "length":[]}
        data.update({f"< {v}":[] for v in volt_range})
    
    # Loop over connection type
    for conn_type in conn_list:
        if conn_type == "nearest":
            # nearest available point algorithm
            for rating in rating_list:
                fx.demand = float(rating * 1000 / 24.0)
                data["lambda"].append(1)
                data["connection"].append("nearest")
                data["rating"].append(rating)
                
                # initial read
                synth_net, evcs = fx.read_inputs()
                init_length = sum([synth_net.edges[e]["length"] \
                                    for e in synth_net.edges])
                
                # additional edges for nearest point routing
                synth_net,opt_flag = fx.connect_evcs(
                    synth_net, evcs, 
                    connection="nearest",
                    epsilon=1e-1,)
                
                # update data dict
                data = update_data(data, synth_net, opt_flag, 
                                   volt_range, init_length)
        
        
    
    
    # optimal routing algorithm
    for lambda_ in lambda_list:
        for rating in rating_list:
            
            print(f"Computing for {fx.area} : lambda={lambda_} : rating={rating}")
            fx.demand = float(rating * 1000 / 24.0)
            
            data["lambda"].append(lambda_)
            data["connection"].append("optimal")
            data["rating"].append(rating)
            
            # initial read
            synth_net, evcs = fx.read_inputs()
            init_length = sum([synth_net.edges[e]["length"] \
                                for e in synth_net.edges])
            
            # additional edges for routing
            synth_net,opt_flag = fx.connect_evcs(
                synth_net, evcs, 
                connection="optimal",
                lambda_ = lambda_, 
                epsilon=1e-1,)
            
            # update data dict
            data = update_data(data, synth_net, opt_flag, 
                               volt_range, init_length)
            
            
    
    # Create the dataframe
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)