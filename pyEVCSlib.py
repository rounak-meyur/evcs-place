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
from matplotlib.lines import Line2D
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
from pyLPSolverlib import get_optimal_routing, cvxpy_solve

sublist = [121143, 121144, 147793, 148717, 148718, 148719, 148720, 148721, 148723,
       150353, 150589, 150638, 150692, 150722, 150723, 150724, 150725, 150726, 
       150727, 150728]

adoption = {
    30: 10, 
    50: 15, 
    100: 30, 
    120: 35, 
    150: 40, 
    180: 50, 
    250: 70, 
    350: 100
}

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
        fig.savefig(to_file, bbox_inches="tight", **kwargs)
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
        self.demand = 30000.0 / 24.0

        self.evfilename = {
            'existing': 'ev-stations',
            'ashik-iter1': 'ev-stations-1'}
        self.area_codes = {f'Area {i+1}': s for i,s in enumerate(sublist)}
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
    def connect_optimal_road(
        net, points, path, 
        epsilon=5e-3, lambda_=1e3, 
        solver="gurobi", verbose=False):
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
        if solver == "gurobi":
            add_edges = get_optimal_routing(
                net, dict(zip(candidates,distance)), points, path, lambda_=lambda_)
        else:
            add_edges = cvxpy_solve(
                net, dict(zip(candidates,distance)), points, 
                lambda_=lambda_, verbose=verbose)
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
                     connection = "nearest", 
                     df_data = None,
                     **kwargs):
        # Check if EVCS exists
        if len(evcs.cord)==0:
            print("No EV charging station exists within region")
            return synth_net
        
        eps = kwargs.get("epsilon", 5e-3)
        lambda_ = kwargs.get("lambda_", 1e3)
        solver = kwargs.get("solver", "gurobi")
        verbose = kwargs.get("verbose", False)
        
        if connection == "nearest":
            new_edges = self.connect_nearest_road(synth_net, evcs, epsilon=eps)
        
        elif connection == "optimal":
            new_edges = self.connect_optimal_road(
                synth_net, evcs, self.grb_dir, 
                epsilon=eps, lambda_=lambda_, 
                solver=solver, verbose=verbose)
            
        else:
            raise ValueError(f"{connection} is invalid connection algorithm specifier")
        
        # Add the attributes
        if new_edges == []:
            return synth_net, 0
        else:
            self.add_attributes(synth_net, new_edges, evcs)
            return synth_net, 1
            
    
    def compute_powerflow(self, net):
        # Run power flow for the network
        powerflow(net)
        return
    
    
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
        plt.box(False)
        
        # ---- Edit the title of the plot ----

        if file_name_sfx := kwargs.get('file_name_sfx'):
            if not to_file:
                to_file = f"{area}-regions"
            to_file = f"{to_file}_{file_name_sfx}"

        if no_ax:
            to_file = f"{self.fig_dir}/{to_file}.png"
            suptitle = f"{self.area}"
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
    
    def plot_investment(
            self, csv_file = None, df_data=None, area=None,
            ax=None, to_file=None, show=True,
            adoptions=None,
            **kwargs
            ):
        kwargs.setdefault('figsize', (15, 15))
        fontsize = kwargs.get('fontsize', 30)
        do_return = kwargs.get('do_return', False)
        label_rotation = kwargs.get('label_rotation', 0)
        if not area:
            area = self.area
            
        if csv_file:
            df_data = pd.read_csv(f"{self.out_dir}/{csv_file}")

        # ---- PLOT ----
        fig, ax, no_ax = get_fig_from_ax(ax, **kwargs)
        
        # investment computation
        cost = 180 / 1609.34
        df_data['cost'] = df_data['length'].apply(lambda x: x*cost)
        df_data['adoption'] = df_data['rating'].apply(lambda x: adoption[x])

        if not adoptions:
            adoptions = df_data.adoption.unique()
        else:
            df_data = df_data.loc[df_data["adoption"].isin(adoptions)]

        sns.barplot(df_data, x="adoption", y="cost", hue="connection",
                    ax=ax, palette=sns.color_palette("Set2"), 
                    edgecolor="k", ci=None)
        
        
        ax.set_xlabel("Percentage EV adoption (%)", fontsize=fontsize)
        ax.set_ylabel("Investment on new lines (1000$)", fontsize=fontsize)
        ax.tick_params(axis='y',labelsize=30)
        ax.tick_params(axis='x',labelsize=30,rotation=label_rotation)
        
        ax.legend(prop={'size': 30},loc='upper left',ncol=2)
        
        # ---- Edit the title of the plot ----

        if file_name_sfx := kwargs.get('file_name_sfx'):
            if not to_file:
                to_file = f"{area}"
            to_file = f"{to_file}_{file_name_sfx}"

        if no_ax:
            to_file = f"{self.fig_dir}/{to_file}.png"
            suptitle = f"{self.area}"
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
            adoptions=None,
            **kwargs
            ):
        
        kwargs.setdefault('figsize', (15, 15))
        fontsize = kwargs.get('fontsize', 30)
        do_return = kwargs.get('do_return', False)
        label_rotation = kwargs.get('label_rotation', 0)
        if not area:
            area = self.area
            
        if csv_file:
            df_data = pd.read_csv(f"{self.out_dir}/{csv_file}")
        
        groups_string = []
        vrange = sorted([float(x.lstrip("< ")) for x in df_data.columns if '<' in x])[::-1]
        for k in range(len(vrange)):
            if k != len(vrange) - 1:
                groups_string.append(f"{vrange[k+1]} - {vrange[k]} pu")
            else:
                groups_string.append(f"< {vrange[k]}")
        groups = [x for x in df_data.columns if '<' in x]
        num_stack = len(groups)
        colors = sns.color_palette("Set3")[:num_stack]
        df_data['adoption'] = df_data['rating'].apply(lambda x: adoption[x])
        
        if not adoptions:
            adoptions = df_data.adoption.unique()
        else:
            df_data = df_data.loc[df_data["adoption"].isin(adoptions)]

        # ---- PLOT ----
        fig, ax, no_ax = get_fig_from_ax(ax, **kwargs)
        for i,g in enumerate(groups):
            ax = sns.barplot(data=df_data, x="adoption", y=g, hue="connection",
                        palette=[colors[i]], ax=ax,
                        zorder=i, edgecolor="k", ci=None)
        
        
        ax.set_xlabel("Percentage EV adoption (%)", fontsize=fontsize)
        ax.set_ylabel("Percentage of nodes within voltage range (%)", fontsize=fontsize)
        ax.tick_params(axis='y',labelsize=30)
        ax.tick_params(axis='x',labelsize=30,rotation=label_rotation)

        hatches = itertools.cycle(['/', ''])
        for i, bar in enumerate(ax.patches):
            if i%(len(adoptions)) == 0:
                hatch = next(hatches)
            bar.set_hatch(hatch)


        han1 = [Patch(facecolor=color, edgecolor='black', label=label) \
                      for label, color in zip(groups_string, colors)]
        han2 = [Patch(facecolor="white",edgecolor='black',
                      label="nearest routing",hatch='/'),
                       Patch(facecolor="white",edgecolor='black',
                             label="optimal routing",hatch='')]
        # leg1 = ax.legend(handles=han1,ncol=1,prop={'size': 50},loc='center right')
        ax.legend(handles=han1+han2,ncol=1,prop={'size': 30},loc='upper left')
        # ax.add_artist(leg1)
        
        # ---- Edit the title of the plot ----

        if file_name_sfx := kwargs.get('file_name_sfx'):
            if not to_file:
                to_file = f"{area}"
            to_file = f"{to_file}_{file_name_sfx}"

        if no_ax:
            to_file = f"{self.fig_dir}/{to_file}.png"
            suptitle = f"{self.area}"
            if suptitle_sfx := kwargs.get('suptitle_sfx'):
                suptitle = f"{suptitle} : {suptitle_sfx}"

            fig.suptitle(suptitle, fontsize=fontsize+3)
            close_fig(fig, to_file, show)

        if do_return:
            return fig, ax
        pass
    
    def plot_tradeoff(
            self, csv_file = None, df_data=None, area=None,
            ax=None, to_file=None, show=True,
            adoptions = None,
            **kwargs,
            ):
        kwargs.setdefault('figsize', (42, 15))
        linewidth = kwargs.get("linewidth", 1)
        markersize = kwargs.get("markersize", 500)
        fontsize = kwargs.get('fontsize', 30)
        tickfontsize = kwargs.get('tick_fontsize', 30)
        do_return = kwargs.get('do_return', False)
        label_rotation = kwargs.get('label_rotation', 0)
        if not area:
            area = self.area
            
        if csv_file:
            df_data = pd.read_csv(f"{self.out_dir}/{csv_file}")
            
        # Filter out only data corresponding to optimal routing
        df_data = df_data.loc[df_data["connection"]=="optimal"]
        
        # ---- PLOT ----
        vrange = [0.92, 0.95, 0.97]
        fig, axs, no_ax = get_fig_from_ax(ax, ndim=(1,len(vrange)), **kwargs)

        # Line plot of undervoltage nodes to investment
        cost = 180 / 1609.34
        df_data['cost'] = df_data['length'].apply(lambda x: x*cost)
        df_data['adoption'] = df_data['rating'].apply(lambda x: adoption[x])

        if not adoptions:
            adoptions = df_data.adoption.unique()
        else:
            df_data = df_data.loc[df_data["adoption"].isin(adoptions)]
        
        colors = sns.color_palette("Set2")

        for i,vr in enumerate(vrange):
            sns.lineplot(
                data = df_data, x = "cost", y = f"< {vr}", 
                hue = "adoption", style = "adoption",
                ax=axs[i], palette=colors, 
                markers = True, 
                lw=linewidth, ms=markersize, 
                )
            
            axs[i].set_xlabel("Investment on new lines (1000 $)", fontsize=fontsize)
            axs[i].set_ylabel("Percentage of undervoltage nodes (%)", fontsize=fontsize)
            axs[i].set_title(f"Reliable voltage limit : {vr} p.u.", fontsize=fontsize)
            axs[i].tick_params(axis='y',labelsize=tickfontsize)
            axs[i].tick_params(axis='x',labelsize=tickfontsize,rotation=label_rotation)

            axs[i].legend(handles=axs[i].get_legend().legendHandles, 
                          labels = [f"{r} %" for r in adoptions], 
                          ncol=1,fontsize=fontsize,loc='upper right', 
                          markerscale=3, 
                          title=f"Percentage \n EV adoption", title_fontsize=fontsize)
        
        
        # ---- Edit the title of the plot ----

        if file_name_sfx := kwargs.get('file_name_sfx'):
            if not to_file:
                to_file = f"{area}"
            to_file = f"{to_file}_{file_name_sfx}"

        if no_ax:
            to_file = f"{self.fig_dir}/{to_file}.png"
            suptitle = f"{self.area}"
            if suptitle_sfx := kwargs.get('suptitle_sfx'):
                suptitle = f"{suptitle} : {suptitle_sfx}"

            fig.suptitle(suptitle, fontsize=fontsize+8)
            close_fig(fig, to_file, show)

        if do_return:
            return fig, ax
        pass
