# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 11:59:15 2022

Author: Rounak Meyur

Description: Contains methods and classes to connect EV charging stations to 
the existing distribution network
"""

import warnings
warnings.filterwarnings("ignore")

import os
import unittest
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import itertools
from timeit import default_timer as timer
from tqdm import tqdm

import csv
import pandas as pd
import networkx as nx
from shapely.geometry import Point, MultiPoint, LineString, base as sg
from collections import namedtuple as nt
import seaborn as sns

from pyUtilslib import GetDistNet, geodist, powerflow
from pyUtilslib import plot_network, highlight_regions
from pyLPSolverlib import get_optimal_routing



def get_fig_from_ax(ax, figsize, **kwargs):
    if not ax:
        no_ax = True
        ndim = kwargs.get('ndim', (1, 1))
        fig, ax = plt.subplots(*ndim, figsize=figsize)
    else:
        no_ax = False
        if not isinstance(ax, matplotlib.axes.Axes):
            getter = kwargs.get('ax_getter', lambda x: x[0])
            ax = getter(ax)
        fig = ax.get_figure()

    return fig, ax, no_ax


def close_fig(fig, to_file=None, show=True, **kwargs):
    if to_file:
        fig.savefig(to_file, **kwargs)
    if show:
        plt.show()
    plt.close(fig)
    pass


def timeit(f, *args, **kwargs):
    start = timer()
    outs = f(*args, **kwargs)
    end = timer()
    return outs, end - start




class EVCSFixture(unittest.TestCase):
    
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.dist_path = "./input/dist-net"
        self.evloc_path = "./input/ev-loc"
        self._out_dir = "out"
        self._fig_dir = "figs"
        self._grb_dir = "gurobi"

        # multiplier for average hourly demand in a charging station in kW
        self.demand = 3600.0 / 24.0

        self.evfilename = {
            'existing': 'ev-stations',
            'ashik-iter1': 'ev-stations-1'}
        self.area_codes = {
            'Area 1': 121144,
            'Area 2': 150692}
        pass
    
    # Out directory setter/ if not, create a directory
    @property
    def out_dir(self):
        return self._out_dir

    @out_dir.setter
    def out_dir(self, out):
        self._out_dir = out
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        pass
    
    # Figures directory setter/ if not, create a directory
    @property
    def fig_dir(self):
        return self._fig_dir

    @fig_dir.setter
    def fig_dir(self, fig):
        self._fig_dir = fig
        if not os.path.exists(self.fig_dir):
            os.makedirs(self.fig_dir)
        pass
    
    @property
    def grb_dir(self):
        return self._grb_dir

    @grb_dir.setter
    def grb_dir(self, grb):
        self._grb_dir = grb
        if not os.path.exists(self.grb_dir):
            os.makedirs(self.grb_dir)
        pass
    
    def read_synthetic_network(self, codes=list(), area=None, hull=None):
        if not area:
            area = self.area
        # Synthetic network
        if not codes:
            codes = self.area_codes[area]

        # Retreive the network data
        synt_net = GetDistNet(self.dist_path, codes)

        # Get the synthetic network edges in the region if a convex hull is provided
        if hull:
            synt_nodes = [
                n for n in synt_net.nodes
                if Point(synt_net.nodes[n]['cord']).within(hull)
                   and synt_net.nodes[n]['label'] != 'H'
            ]
            synt_net = nx.subgraph(synt_net, synt_nodes)

        return synt_net
    
    def read_ev_data(self, filename, hull=None):
        if not filename.endswith(".csv"):
            filename = f"{filename}.csv"
        df = pd.read_csv(f"{self.evloc_path}/{filename}", header=0)

        if hull:
            fuel = [(df["x_station"][i],df["y_station"][i], df["client_count"][i]) \
                    for i in range(len(df)) \
                    if Point(df["x_station"][i],df["y_station"][i]).within(hull)]
        else:
            fuel = [(df["x_station"][i],df["y_station"][i], df["client_count"][i]) \
                    for i in range(len(df))]
        
        # Rename the fuel station nodes
        evcs_cord = {f'F{i+1}': (fuel[i][0],fuel[i][1]) \
                     for i,data in enumerate(fuel)}
        evcs_demand = {f'F{i+1}': fuel[i][-1] * self.demand for i,data in enumerate(fuel)}

        # Store data in named tuple
        evcs = nt('EVCS', field_names=["cord","demand"])
        evcs_data = evcs(cord=evcs_cord, demand=evcs_demand)
        return evcs_data
    
    def read_fuel_data(self, evcsdataID=None, hull=None):
        if not evcsdataID:
            evcsdataID = self.evcsdataID
        
        # Check if data identifier exists in the list of accepted identifiers
        if evcsdataID not in self.evfilename:
            raise ValueError(f"{evcsdataID} not in accepted identifier list!!!")
            
        # EV charging stations data
        evloc_file = f"{self.evloc_path}/{self.evfilename[evcsdataID]}.csv"
        if not os.path.exists(evloc_file):
            raise ValueError(f"{evloc_file} doesn't exist!")
        
        # Load the alternative fuel station data
        df = pd.read_csv(evloc_file, 
                             usecols=["Longitude", "Latitude"])
        df = df.rename(columns={"Longitude":"x","Latitude":"y"})
        
        # Find alternate fuel stations within convex hull
        if hull:
            fuel = [(df["x"][i],df["y"][i], self.demand) for i in range(len(df)) \
                    if Point(df["x"][i],df["y"][i]).within(hull)]
        else:
            fuel = [(df["x"][i],df["y"][i], self.demand) for i in range(len(df))]
        
        # Rename the fuel station nodes
        evcs_cord = {f'F{i+1}': (fuel[i][0],fuel[i][1]) \
                     for i,data in enumerate(fuel)}
        evcs_demand = {f'F{i+1}': fuel[i][-1] for i,data in enumerate(fuel)}
        
        # Store data in named tuple
        evcs = nt('EVCS', field_names=["cord","demand"])
        evcs_data = evcs(cord=evcs_cord, demand=evcs_demand)
        return evcs_data
    
    def read_fuel_data_near_network(self, dist_net, evcsdataID=None):
        # Convex hull for points in the network forming the enclosed area
        pt_nodes = MultiPoint([Point(dist_net.nodes[n]["cord"]) \
                               for n in dist_net])
        hull = pt_nodes.convex_hull
        
        # Get the EVCS stations within the region
        # evcs = self.read_fuel_data(evcsdataID=evcsdataID, hull=hull)
        evcs = self.read_ev_data("evcs_solution.csv", hull=hull)
        return evcs
    
    def read_inputs(self, area=None, hull=None, evcsdataID=None):

        # Synthetic distribution network data
        dist_net =  self.read_synthetic_network(area=area, hull=hull)
        
        # Get the EVCS stations near the network
        evcs = self.read_fuel_data_near_network(
            dist_net, evcsdataID=evcsdataID)
        return dist_net, evcs
    
    
    @staticmethod
    def connect_nearest_road(net, points, epsilon=5e-3):
        add_edges = []
        for p_node,p_cord in points.cord.items():
            region = Point(p_cord).buffer(epsilon)
            near_road = [n for n in net if net.nodes[n]['label']=='R' and \
                         Point(net.nodes[n]['cord']).within(region)]
            r_dist = {r:geodist(net.nodes[r]['cord'],p_cord) for r in near_road}
            r_min = min(r_dist, key=r_dist.get)
            add_edges.append((r_min,p_node))
        return add_edges
    
    @staticmethod
    def connect_optimal_road(net, points, path, epsilon=5e-3, lambda_=1e3):
        # Get candidate edges
        candidates = []
        distance = []
        for p_node, p_cord in points.cord.items():
            region = Point(p_cord).buffer(epsilon)
            near_road = [n for n in net if net.nodes[n]['label']=='R' and \
                         Point(net.nodes[n]['cord']).within(region)]
            candidates.extend([(p_node,r) for r in near_road])
            distance.extend([geodist(p_cord,net.nodes[r]['cord']) for r in near_road])
        
        # Solve optimization problem to get the best candidate
        add_edges = get_optimal_routing(
            net, dict(zip(candidates,distance)), points, path, lambda_=lambda_)
        return add_edges
    
    @staticmethod
    def add_attributes(net, new_edges, new_nodes):
        # add the new edges
        net.add_edges_from(new_edges)
        
        # add node attributes
        for node in new_nodes.cord:
            if node in net:
                net.nodes[node]["cord"] = new_nodes.cord[node]
                net.nodes[node]["label"] = "E"
                net.nodes[node]["load"] = new_nodes.demand[node]
        
        # add edge attributes
        for edge in new_edges:
            geom = LineString((net.nodes[edge[0]]["cord"],
                               net.nodes[edge[1]]["cord"]))
            l = geodist(net.nodes[edge[0]]["cord"],net.nodes[edge[1]]["cord"])
            net.edges[edge]["label"] = "L"
            net.edges[edge]["length"] = l
            net.edges[edge]["r"] = 0.0822/39690 * l * 1e-3
            net.edges[edge]["x"] = 0.0964/39690 * l * 1e-3
            net.edges[edge]["type"] = "OH_Penguin"
            net.edges[edge]["geometry"] = geom
            
            # add the reach attribute to the node
            if net.nodes[edge[0]]['label'] == 'E':
                net.nodes[edge[0]]['reach'] = net.nodes[edge[1]]['reach'] + l
            elif net.nodes[edge[1]]['label'] == 'E':
                net.nodes[edge[1]]['reach'] = net.nodes[edge[0]]['reach'] + l
            else:
                raise ValueError(f"{edge} does not have an EVCS node")
        return
    
    def connect_evcs(self, synth_net, evcs, 
                     connection = "nearest", **kwargs):
        # Check if EVCS exists
        if len(evcs.cord)==0:
            print("No EV charging station exists within region")
            return synth_net
        
        eps = kwargs.get("epsilon", 5e-3)
        lambda_ = kwargs.get("lambda_", 1e3)
        
        if connection == "nearest":
            new_edges = self.connect_nearest_road(synth_net, evcs, epsilon=eps)
        
        elif connection == "optimal":
            new_edges = self.connect_optimal_road(
                synth_net, evcs, self.grb_dir, epsilon=eps, lambda_=lambda_)
            
        else:
            raise ValueError(f"{connection} is invalid connection algorithm specifier")
        
        # Add the attributes
        self.add_attributes(synth_net, new_edges, evcs)
        return synth_net
    
    def compute_powerflow(self, net):
        # Run power flow for the network
        powerflow(net)
        return
    
    # def plot_voltage(self,net,ax):
    #     color_ = {"E":"blue", "P":"black", "S":"red", "L":"peru"}
    #     v_nodes = {n:net.nodes[n]["voltage"] for n in net}
    #     for edge in net.edges:
    #         ax.plot([d_nodes[edge[0]],d_nodes[edge[1]]],
    #                   [v_nodes[edge[0]],v_nodes[edge[1]]],
    #                   color=color_[net.edges[edge]["label"]])
    
    
    # def plot_voltage_comparison(
    #         self, net, demand,
    #         ax=None, to_file=None, show=True,
    #         **kwargs
    #         ):
        
    #     # ---- Arguments ----
    #     fontsize = kwargs.get('fontsize', 25)
    #     do_return = kwargs.get('do_return', False)
    #     figsize = kwargs.get('figsize', (20, 10))
    #     fig, axs, no_ax = get_fig_from_ax(ax, figsize, ndim=(1, 2))

    #     # ---- PLOT ----
    #     compute_powerflow(net)
        
    #     # ---- Save plot figure ----
    #     if no_ax:
    #         to_file = f"{self.fig_dir}/{to_file}.png"
    #         if kwargs.get('suptitle_sfx'):
    #             suptitle = f"{kwargs.get('suptitle_sfx')}"
    #             fig.suptitle(suptitle,fontsize=fontsize)
    #         close_fig(fig, to_file, show, bbox_inches='tight')

    #     if do_return:
    #         return fig, axs
    #     pass
    
    def plot_synth_net(
            self, synth_net, area=None,
            ax=None, to_file=None, show=True,
            **kwargs
            ):
        kwargs.setdefault('figsize', (40, 20))
        fontsize = kwargs.get('fontsize', 20)
        do_return = kwargs.get('do_return', False)
        if not area:
            area = self.area

        # ---- PLOT ----
        fig, ax, no_ax = get_fig_from_ax(ax, **kwargs)
        plot_network(synth_net, ax, **kwargs)
        
        # ---- Edit the title of the plot ----

        if file_name_sfx := kwargs.get('file_name_sfx'):
            if not to_file:
                to_file = f"{area}-regions"
            to_file = f"{to_file}_{file_name_sfx}"

        if no_ax:
            to_file = f"{self.fig_dir}/{to_file}.png"
            suptitle = f"${self.area}$"
            if suptitle_sfx := kwargs.get('suptitle_sfx'):
                suptitle = f"{suptitle} : {suptitle_sfx}"

            fig.suptitle(suptitle, fontsize=fontsize+3)
            close_fig(fig, to_file, show)

        if do_return:
            return fig, ax
        pass
    
    
    def plot_evcs_region(
            self, synth_net, region_list, area=None,
            ax=None, to_file=None, show=True,
            **kwargs
            ):
        kwargs.setdefault('figsize', (40, 20))
        fontsize = kwargs.get('fontsize', 20)
        do_return = kwargs.get('do_return', False)
        if not area:
            area = self.area

        # ---- PLOT ----
        fig, ax, no_ax = get_fig_from_ax(ax, **kwargs)
        plot_network(synth_net, ax, **kwargs)
        
        # ---- PLOT REGIONS ----
        highlight_regions(region_list, ax, **kwargs)
        
        # ---- Edit the title of the plot ----

        if file_name_sfx := kwargs.get('file_name_sfx'):
            if not to_file:
                to_file = f"{area}-regions"
            to_file = f"{to_file}_{file_name_sfx}"

        if no_ax:
            to_file = f"{self.fig_dir}/{to_file}.png"
            suptitle = f"${self.area}$"
            if suptitle_sfx := kwargs.get('suptitle_sfx'):
                suptitle = f"{suptitle} : {suptitle_sfx}"

            fig.suptitle(suptitle, fontsize=fontsize+3)
            close_fig(fig, to_file, show)

        if do_return:
            return fig, ax
        pass
    
    def plot_dependence(
            self, csv_file = None, df_data=None, area=None,
            ax=None, to_file=None, show=True,
            **kwargs
            ):
        kwargs.setdefault('figsize', (10, 10))
        fontsize = kwargs.get('fontsize', 30)
        do_return = kwargs.get('do_return', False)
        if not area:
            area = self.area
            
        if csv_file:
            df_data = pd.read_csv(f"{self.out_dir}/{csv_file}")

        # ---- PLOT ----
        fig, ax, no_ax = get_fig_from_ax(ax, **kwargs)
        
        sns.barplot(df_data, x="rating", y="length", hue="connection",
                    ax=ax)
        
        
        ax.set_xlabel("EV charger rating (Watts)", fontsize=fontsize)
        ax.set_ylabel("Additional length (in meters)", fontsize=fontsize)
        ax.tick_params(axis='y',labelsize=30)
        ax.tick_params(axis='x',labelsize=30,rotation=60)
        
        ax.legend(prop={'size': 30},loc='center left')
        
        # ---- Edit the title of the plot ----

        if file_name_sfx := kwargs.get('file_name_sfx'):
            if not to_file:
                to_file = f"{area}"
            to_file = f"{to_file}_{file_name_sfx}"

        if no_ax:
            to_file = f"{self.fig_dir}/{to_file}.png"
            suptitle = f"${self.area}$"
            if suptitle_sfx := kwargs.get('suptitle_sfx'):
                suptitle = f"{suptitle} : {suptitle_sfx}"

            fig.suptitle(suptitle, fontsize=fontsize+3)
            close_fig(fig, to_file, show)

        if do_return:
            return fig, ax
        pass
    
    def plot_improvement(
            self, csv_file = None, df_data=None, area=None,
            ax=None, to_file=None, show=True,
            **kwargs
            ):
        
        kwargs.setdefault('figsize', (10, 10))
        fontsize = kwargs.get('fontsize', 30)
        do_return = kwargs.get('do_return', False)
        if not area:
            area = self.area
            
        if csv_file:
            df_data = pd.read_csv(f"{self.out_dir}/{csv_file}")
        
        cols = df_data.columns
        groups = [x for x in cols if '<' in x]
        num_stack = len(groups)
        colors = sns.color_palette("Set3")[:num_stack]
        ratings = df_data.rating.unique()

        # ---- PLOT ----
        fig, ax, no_ax = get_fig_from_ax(ax, **kwargs)
        for i,g in enumerate(groups):
            ax = sns.barplot(data=df_data, x="rating", y=g, hue="connection",
                        palette=[colors[i]], ax=ax,
                        zorder=i, edgecolor="k")
        
        
        ax.set_xlabel("EV charger rating (Watts)", fontsize=fontsize)
        ax.set_ylabel("Number of nodes", fontsize=fontsize)
        ax.tick_params(axis='y',labelsize=30)
        ax.tick_params(axis='x',labelsize=30,rotation=60)

        hatches = itertools.cycle(['/', ''])
        for i, bar in enumerate(ax.patches):
            if i%(len(ratings)) == 0:
                hatch = next(hatches)
            bar.set_hatch(hatch)


        han1 = [Patch(facecolor=color, edgecolor='black', label=label) \
                      for label, color in zip(groups, colors)]
        han2 = [Patch(facecolor="white",edgecolor='black',
                      label="optimal routing",hatch='/'),
                       Patch(facecolor="white",edgecolor='black',
                             label="nearest routing",hatch='')]
        # leg1 = ax.legend(handles=han1,ncol=1,prop={'size': 50},loc='center right')
        ax.legend(handles=han1+han2,ncol=1,prop={'size': 20},loc='upper left')
        # ax.add_artist(leg1)
        
        # ---- Edit the title of the plot ----

        if file_name_sfx := kwargs.get('file_name_sfx'):
            if not to_file:
                to_file = f"{area}"
            to_file = f"{to_file}_{file_name_sfx}"

        if no_ax:
            to_file = f"{self.fig_dir}/{to_file}.png"
            suptitle = f"${self.area}$"
            if suptitle_sfx := kwargs.get('suptitle_sfx'):
                suptitle = f"{suptitle} : {suptitle_sfx}"

            fig.suptitle(suptitle, fontsize=fontsize+3)
            close_fig(fig, to_file, show)

        if do_return:
            return fig, ax
        pass




# class EVCSRuns_Montgomery(EVCSFixture):
    
    
    # def __init__(self, methodName:str = ...) -> None:
    #     super().__init__(methodName)
    #     self.out_dir = "out/script"
    #     self.fig_dir = "figs/script"
    #     self.grb_dir = "gurobi/script"
    #     self.area = "Area 2"
    #     self.evcsdataID = None
    #     self.demand = 3600.0 / 24.0
    #     return
    
    # def test_read_synthetic_network(self):
    #     self.area = 'Area 1'

    #     # read synthetic network
    #     synth_net = self.read_synthetic_network()
    #     self.assertIsNotNone(synth_net)

    #     # plot network
    #     fig, ax = self.plot_synth_net(
    #         synth_net, self.area,
    #         file_name_sfx="synth_net",
    #         do_return=True
    #     )
    #     self.assertIsNotNone(fig)
    #     self.assertIsNotNone(ax)
    #     pass
    
    # def test_read_fuel_data(self):
    #     self.evcsdataID = 'existing'
        
    #     # read fuel input
    #     evcs_data = self.read_fuel_data()
    #     self.assertIsNotNone(evcs_data)
    #     self.assertIsNotNone(evcs_data.cord)
    #     self.assertIsNotNone(evcs_data.demand)
    #     pass
    
    # def test_read_inputs(self):
    #     self.evcsdataID = 'existing'
    #     self.area = 'Area 2'
        
    #     # Read fuel input
    #     synth_net, evcs_data = self.read_inputs()
    #     self.assertIsNotNone(synth_net)
    #     self.assertIsNotNone(evcs_data)
    #     self.assertIsNotNone(evcs_data.cord)
    #     self.assertIsNotNone(evcs_data.demand)
        
    #     # plot network
    #     fig, ax = self.plot_synth_net(
    #         synth_net, self.area,
    #         title_sfx = "Synthetic power distribution network", fontsize=30,
    #         file_name_sfx="synth_net",
    #         do_return=True
    #     )
    #     self.assertIsNotNone(fig)
    #     self.assertIsNotNone(ax)
    #     pass
    
    # def test_connect_evcs(self):
    #     # self.evcsdataID = 'existing'
    #     self.area = 'Area 2'
        
    #     # Test the function
    #     # if no information is available, use default setting
    #     synth_net, evcs = self.read_inputs()
    #     synth_net = self.connect_evcs(synth_net, evcs)
    #     self.assertIsNotNone(synth_net)
        
    #     # if synthetic network is available but not EVCS information
    #     # synth_net = self.read_synthetic_network()
    #     # synth_net = self.connect_evcs(synth_net=synth_net)
        
    #     # if both information are available
    #     # synth_net = self.connect_evcs(
    #     #     synth_net=synth_net, evcs=evcs_data)
        
    #     # powerflow
    #     powerflow(synth_net)
    #     ev_nodes = [n for n in synth_net if synth_net.nodes[n]['label']=='E']
    #     print(ev_nodes)
    #     print([synth_net.nodes[n]["voltage"] for n in ev_nodes])
        
    #     # plot network
    #     fig, ax = self.plot_synth_net(
    #         synth_net, self.area,
    #         title_sfx = "Power distribution network with EVCS connected to nearest point",
    #         file_name_sfx="synth_net_evcs", fontsize=30,
    #         do_return=True
    #     )
    #     self.assertIsNotNone(fig)
    #     self.assertIsNotNone(ax)
    #     pass
    
    # def test_connect_evcs_optimal(self):
    #     # self.evcsdataID = 'existing'
    #     self.area = 'Area 2'
    #     self.demand = 3600.0 / 24.0
    #     self.connection = 'optimal'
        
    #     synth_net, evcs = self.read_inputs()
    #     self.assertIsNotNone(synth_net)
    #     self.assertIsNotNone(evcs)
    #     self.assertIsNotNone(evcs.cord)
    #     self.assertIsNotNone(evcs.demand)
        
    #     synth_net = self.connect_evcs(
    #         synth_net, evcs, 
    #         connection=self.connection,
    #         lambda_ = 1e3, 
    #         epsilon = 1e-2,)
    #     self.assertIsNotNone(synth_net)
        
    #     # plot network
    #     fig, ax = self.plot_synth_net(
    #         synth_net,
    #         suptitle_sfx = f"EVCS connected to {self.connection} point : rating {self.demand*24}W",
    #         file_name_sfx = f"synth_net_evcs_{self.demand}kw_{self.connection}_ashik_data2", 
    #         fontsize=30,
    #         do_return=True
    #     )
    #     self.assertIsNotNone(fig)
    #     self.assertIsNotNone(ax)
    #     pass
    
    # def test_demand_dependence(self):
    #     # self.evcsdataID = 'existing'
    #     self.area = 'Area 2'
        
    #     # initial read
    #     synth_net, evcs = self.read_inputs()
    #     init_length = sum([synth_net.edges[e]["length"] \
    #                         for e in synth_net.edges])
        
    #     volt_range = [0.95, 0.92, 0.90]
    #     data = {"rating":[], "connection":[], "length":[]}
    #     data.update(
    #         {f"less than {v}":[] for v in volt_range}
    #         )
    #     lambda_ = 1000000
    #     rating_list = [1800, 2000, 2400, 3000, 3600, 4800]
        
    #     for conn_type in ["optimal", "nearest"]:
        
    #         for k in tqdm(range(len(rating_list)), 
    #                       desc="Computing for different EV charger ratings",
    #                       ncols=100):
    #             self.demand = float(rating_list[k] / 24.0)
                
    #             # additional edges for routing
    #             synth_net = self.connect_evcs(
    #                 synth_net, evcs, 
    #                 connection=conn_type,
    #                 lambda_ = lambda_, 
    #                 epsilon=1e-1,)
    #             final_length = sum([synth_net.edges[e]["length"] \
    #                                 for e in synth_net.edges])
                
    #             # Evaluate the additional length
    #             add_length = final_length - init_length
                
    #             # Add it to the data
    #             data["connection"].append(conn_type)
    #             data["rating"].append(rating_list[k])
    #             data["length"].append(add_length)
                
    #             # run powerflow and number of nodes outside limit
    #             powerflow(synth_net)
    #             nodelist = [n for n in synth_net if synth_net.nodes[n]['label']!='R']
    #             for v in volt_range:
    #                 num_nodes = len([n for n in nodelist if synth_net.nodes[n]["voltage"] < v])
    #                 data[f"less than {v}"].append(num_nodes)
        
    #     # Create the dataframe
    #     df = pd.DataFrame(data)
    #     df.to_csv(f"{self.out_dir}/demand_lamb_{lambda_}.csv", index=False)
        
    #     # Plot the dependence
    #     fig, ax = self.plot_dependence(
    #         df_data=df,
    #         suptitle_sfx = f"Additional network length versus demand : lambda = {lambda_}",
    #         file_name_sfx = f"demand_dependence_lamb_{lambda_}", 
    #         fontsize=20,
    #         do_return=True
    #     )
    #     self.assertIsNotNone(fig)
    #     self.assertIsNotNone(ax)
        
    #     pass
    
    # def test_plot_dependence(self):
    #     lambda_ = 1000000
    #     fig, ax = self.plot_dependence(
    #         csv_file = f"demand_lamb_{lambda_}.csv",
    #         suptitle_sfx = f"Additional network length versus demand : lambda = {lambda_}",
    #         file_name_sfx = f"demand_dependence_lamb_{lambda_}", 
    #         fontsize=30,
    #         do_return=True
    #     )
    #     self.assertIsNotNone(fig)
    #     self.assertIsNotNone(ax)
    #     pass
    
    # def test_lambda_optimal_dependence(self):
    #     self.evcsdataID = 'existing'
    #     self.area = 'Area 2'
    #     self.demand = 2000
        
    #     synth_net, evcs = self.read_inputs()
    #     init_length = sum([synth_net.edges[e]["length"] \
    #                        for e in synth_net.edges])
        
    #     data_lambda = {"lambda":[], "length":[]}
        
    #     for lambda_ in tqdm(range(0,1000000,50000), 
    #                   ncols=100, desc="simulate for lambdas"):
            
    #         synth_net = self.connect_evcs(
    #             synth_net, evcs, 
    #             connection='optimal',
    #             lambda_ = lambda_, 
    #             epsilon=1e-1,)
    #         final_length = sum([synth_net.edges[e]["length"] \
    #                            for e in synth_net.edges])
            
    #         # Evaluate the additional length
    #         add_length = final_length - init_length
            
    #         # Add it to the data
    #         data_lambda["lambda"].append(lambda_)
    #         data_lambda["length"].append(add_length)
        
    #     # Create the dataframe
    #     df = pd.DataFrame(data_lambda)
    #     df.to_csv(f"{self.out_dir}/lambda.csv", index=False)
    #     return
            
            
    
    
    # def test_show_region(self):
    #     self.evcsdataID = 'existing'
    #     self.area = 'Area 2'
        
    #     # Read fuel input
    #     synth_net, evcs_data = self.read_inputs()
    #     self.assertIsNotNone(synth_net)
    #     self.assertIsNotNone(evcs_data)
    #     self.assertIsNotNone(evcs_data.cord)
    #     self.assertIsNotNone(evcs_data.demand)
        
    #     # Get regions
    #     epsilon = 2e-3
    #     region_list = []
    #     for p_node, p_cord in evcs_data.cord.items():
    #         region_list.append(Point(p_cord).buffer(epsilon))
        
    #     # plot network
    #     self.fig_dir = "figs/test"
    #     fig, ax = self.plot_evcs_region(
    #         synth_net, region_list,
    #         alpha = 0.5, region_color='xkcd:purple',
    #         title_sfx = "Network with EVCS and nearby region",
    #         file_name_sfx="evcs_region", fontsize=30,
    #         do_return=True
    #     )
    #     self.assertIsNotNone(fig)
    #     self.assertIsNotNone(ax)
    #     pass


# if __name__ == '__main__':
#     unittest.main()




fx = EVCSFixture('runTest')
fx.out_dir = "out/script"
fx.fig_dir = "figs/script"
fx.grb_dir = "gurobi/script"
fx.area = 'Area 2'

fx.plot_dependence(csv_file="demand_lamb_1000000.csv", 
                    suptitle_sfx = "additional length for routing power lines")

fx.plot_improvement(csv_file="demand_lamb_1000000.csv", 
                    suptitle_sfx = "improvement in voltages")

import sys
sys.exit(0)


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
            data[f"< {v}"].append(num_nodes)

# Create the dataframe
df = pd.DataFrame(data)
df.to_csv(f"{fx.out_dir}/demand_lamb_{lambda_}.csv", index=False)

# Plot the dependence
fig, ax = fx.plot_dependence(
    df_data=df,
    suptitle_sfx = f"Additional network length versus demand : lambda = {lambda_}",
    file_name_sfx = f"demand_dependence_lamb_{lambda_}", 
    fontsize=20,
    do_return=True
)
fx.assertIsNotNone(fig)
fx.assertIsNotNone(ax)












































