import os
import networkx as nx
import numpy as np
from math import log,exp
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from shapely.geometry import Point, LineString
from geographiclib.geodesic import Geodesic
import pickle
from networkx.utils import open_file

@open_file(0, mode='rb')
def read_gpickle(path):
    return pickle.load(path)

def geodist(geomA,geomB):
    if type(geomA) != Point: geomA = Point(geomA)
    if type(geomB) != Point: geomB = Point(geomB)
    geod = Geodesic.WGS84
    return geod.Inverse(geomA.y, geomA.x, geomB.y, geomB.x)['s12']

def GetDistNet(path,code):
    """
    Read the txt file containing the edgelist of the generated synthetic network and
    generates the corresponding networkx graph. The graph has the necessary node and
    edge attributes.
    
    Inputs:
        path: name of the directory
        code: substation ID or list of substation IDs
        
    Output:
        graph: networkx graph
        node attributes of graph:
            cord: longitude,latitude information of each node
            label: 'H' for home, 'T' for transformer, 'R' for road node, 
                    'S' for subs
            voltage: node voltage in pu
        edge attributes of graph:
            label: 'P' for primary, 'S' for secondary, 'E' for feeder lines
            r: resistance of edge
            x: reactance of edge
            geometry: shapely geometry of edge
            geo_length: length of edge in meters
            flow: power flowing in kVA through edge
    """
    if type(code) == list:
        graph = nx.Graph()
        for c in code:
            net_file = f"{path}/{c}-dist-net.gpickle"
            if not os.path.exists(net_file):
                raise ValueError(f"{net_file} doesn't exist!")
            g = read_gpickle(net_file)
            graph = nx.compose(graph,g)
            for n in graph:
                graph.nodes[n]["reach"] = nx.shortest_path_length(
                    graph, n, c, 'length')
    else:
        net_file = f"{path}/{code}-dist-net.gpickle"
        if not os.path.exists(net_file):
            raise ValueError(f"{net_file} doesn't exist!")
        graph = read_gpickle(net_file)
        for n in graph:
            graph.nodes[n]["reach"] = nx.shortest_path_length(
                graph, n, code, 'length')
    return graph


#%% Functions to display networks
def DrawNodes(synth_graph,ax,label=['S','T','H'],color='green',size=25,
              alpha=1.0):
    """
    Get the node geometries in the network graph for the specified node label.
    """
    # Get the nodes for the specified label
    if label == []:
        nodelist = list(synth_graph.nodes())
    else:
        nodelist = [n for n in synth_graph.nodes() \
                    if synth_graph.nodes[n]['label']==label \
                        or synth_graph.nodes[n]['label'] in label]
    # Get the dataframe for node and edge geometries
    if len(nodelist) == 0:
        return
    d = {'nodes':nodelist,
         'geometry':[Point(synth_graph.nodes[n]['cord']) for n in nodelist]}
    df_nodes = gpd.GeoDataFrame(d, crs="EPSG:4326")
    df_nodes.plot(ax=ax,color=color,markersize=size,alpha=alpha)
    return

def DrawEdges(synth_graph,ax,label=['P','E','S'],color='black',width=2.0,
              style='solid',alpha=1.0):
    """
    """
    # Get the nodes for the specified label
    if label == []:
        edgelist = list(synth_graph.edges())
    else:
        edgelist = [e for e in synth_graph.edges() \
                    if synth_graph[e[0]][e[1]]['label']==label\
                        or synth_graph[e[0]][e[1]]['label'] in label]
    if len(edgelist) == 0:
        return
    d = {'edges':edgelist,
         'geometry':[synth_graph.edges[e]['geometry'] for e in edgelist]}
    df_edges = gpd.GeoDataFrame(d, crs="EPSG:4326")
    df_edges.plot(ax=ax,edgecolor=color,linewidth=width,linestyle=style,alpha=alpha)
    return

def draw_polygons(ax, polygons, color='red', alpha=1.0, label=None):
    if len(polygons) == 0:
        return ax
    if isinstance(polygons, list):
        d = {'nodes': range(len(polygons)),
             'geometry': [geom for geom in polygons]}
    elif isinstance(polygons, dict):
        d = {'nodes': range(len(polygons)),
             'geometry': [polygons[k] for k in polygons]}
    df_polygons = gpd.GeoDataFrame(d, crs="EPSG:4326")
    df_polygons.plot(ax=ax, facecolor=color, alpha=alpha, label=label)
    return ax

def highlight_regions(geom_regions, ax,  **kwargs):
    # Index for highlight among the regions
    highlight = kwargs.get('region_highlight', None)
    highlight = kwargs.get('highlight', highlight)
    
    # Transparency level
    alpha = kwargs.get('alpha', 0.2)
    
    ## region color and highlight color
    plot_colors = dict(
        region=kwargs.get('region_color', 'cyan'),
        highlight=kwargs.get('highlight_color', 'orange')
    )
    
    # highlight one region whose index is passed
    if highlight is not None:
        geom_regions = geom_regions[:]
        region = geom_regions.pop(highlight)
        draw_polygons(ax, [region], color=plot_colors['highlight'], 
                      alpha=alpha)
    
    # show the other regions
    draw_polygons(ax, geom_regions, color=plot_colors['region'], alpha=alpha)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    return ax

def plot_network(net, ax, **kwargs):
    """
    """
    sub_size = kwargs.get("substation_size", 2000)
    tsf_size = kwargs.get("transformer_size", 70)
    road_size = kwargs.get("roadnode_size", 1)
    ev_size = kwargs.get("evcs_size", 700)
    res_size = kwargs.get("home_size", 30)
    primnet_wdth = kwargs.get("primnet_width", 2)
    hvnet_wdth = kwargs.get("hvnet_width", 2)
    secnet_wdth = kwargs.get("secnet_width", 1)
    evnet_wdth = kwargs.get("evcs_conn_width", 3)
    fontsize = kwargs.get("fontsize", 30)
    
    # Draw nodes
    DrawNodes(net,ax,label='S',color='dodgerblue',size=sub_size)
    DrawNodes(net,ax,label='T',color='green',size=tsf_size)
    DrawNodes(net,ax,label='R',color='black',size=road_size)
    DrawNodes(net,ax,label='E',color='peru',size=ev_size)
    DrawNodes(net,ax,label='H',color='crimson',size=res_size)
    
    # Draw edges
    DrawEdges(net,ax,label='P',color='black',width=primnet_wdth)
    DrawEdges(net,ax,label='E',color='dodgerblue',width=hvnet_wdth)
    DrawEdges(net,ax,label='L',color='peru',width=evnet_wdth)
    DrawEdges(net,ax,label='S',color='crimson',width=secnet_wdth)
    
    # Legend for the plot
    leghands = [Line2D([0], [0], color='black', markerfacecolor='black', 
                   marker='o',markersize=0,label='primary network'),
                Line2D([0], [0], color='crimson', markerfacecolor='crimson', 
                   marker='o',markersize=0,label='secondary network'),
                Line2D([0], [0], color='dodgerblue', 
                   markerfacecolor='dodgerblue', marker='o',
                   markersize=0,label='high voltage feeder'),
                Line2D([0], [0], color='white', markerfacecolor='green', 
                   marker='o',markersize=20,label='transformer'),
                Line2D([0], [0], color='white', markerfacecolor='dodgerblue', 
                   marker='o',markersize=20,label='substation'),
               Line2D([0], [0], color='white', markerfacecolor='red', 
                   marker='o',markersize=20,label='residence')]
    # If EV charging station present add the legend
    if len([n for n in net if net.nodes[n]['label']=='E']) != 0:
        leghands.append(
            Line2D([0], [0], color='white', markerfacecolor='peru', 
               marker='o',markersize=20,label="EV charging station")
            )
        leghands.insert(2,
            Line2D([0], [0], color='peru', markerfacecolor='peru', 
               marker='o',markersize=0,label='EVCS connection')
            )
    
    ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
    ax.legend(handles=leghands,loc='best',ncol=2,
              fontsize=fontsize)
    return


#%% Power flow problem
def powerflow(graph,v0=1.0):
    """
    Checks power flow solution and save dictionary of voltages.
    """
    # Pre-processing to rectify incorrect code
    hv_lines = [e for e in graph.edges if graph.edges[e]['label']=='E']
    for e in hv_lines:
        try:
            length = graph.edges[e]['length']
        except:
            length = graph.edges[e]['geo_length']
        graph.edges[e]['r'] = (0.0822/363000)*length*1e-3
        graph.edges[e]['x'] = (0.0964/363000)*length*1e-3
    
    # Main function begins here
    A = nx.incidence_matrix(graph,nodelist=list(graph.nodes()),
                            edgelist=list(graph.edges()),oriented=True).toarray()
    
    node_ind = [i for i,node in enumerate(graph.nodes()) \
                if graph.nodes[node]['label'] != 'S']
    nodelist = [node for node in list(graph.nodes()) \
                if graph.nodes[node]['label'] != 'S']
    edgelist = [edge for edge in list(graph.edges())]
    
    # Resistance data
    edge_r = []
    for e in graph.edges:
        try:
            edge_r.append(1.0/graph.edges[e]['r'])
        except:
            edge_r.append(1.0/1e-14)
    R = np.diag(edge_r)
    G = np.matmul(np.matmul(A,R),A.T)[node_ind,:][:,node_ind]
    p = np.array([1e-3*graph.nodes[n]['load'] for n in nodelist])
    
    # Voltages and flows
    v = np.matmul(np.linalg.inv(G),p)
    f = np.matmul(np.linalg.inv(A[node_ind,:]),p)
    voltage = {n:v0-v[i] for i,n in enumerate(nodelist)}
    flows = {e:log(abs(f[i])+1e-10) for i,e in enumerate(edgelist)}
    subnodes = [node for node in list(graph.nodes()) \
                if graph.nodes[node]['label'] == 'S']
    for s in subnodes: voltage[s] = v0
    nx.set_node_attributes(graph,voltage,'voltage')
    nx.set_edge_attributes(graph,flows,'flow')
    return