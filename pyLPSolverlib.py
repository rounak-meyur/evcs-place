# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 10:37:07 2022

Author: Rounak Meyur
"""

import sys
import gurobipy as grb
import numpy as np
import scipy
import networkx as nx
import cvxpy as cp

def mycallback(model, where):
    if where == grb.GRB.Callback.MIP:
        # General MIP callback
        objbst = model.cbGet(grb.GRB.Callback.MIP_OBJBST)
        objbnd = model.cbGet(grb.GRB.Callback.MIP_OBJBND)
        time = model.cbGet(grb.GRB.Callback.RUNTIME)
        if(time>300 and abs(objbst - objbnd) < 0.005 * (1.0 + abs(objbst))):
            print('Stop early - 0.50% gap achieved time exceeds 5 minutes')
            model.terminate()
        elif(time>60 and abs(objbst - objbnd) < 0.0025 * (1.0 + abs(objbst))):
            print('Stop early - 0.25% gap achieved time exceeds 1 minute')
            model.terminate()
        elif(time>300 and abs(objbst - objbnd) < 0.01 * (1.0 + abs(objbst))):
            print('Stop early - 1.00% gap achieved time exceeds 5 minutes')
            model.terminate()
        elif(time>480 and abs(objbst - objbnd) < 0.05 * (1.0 + abs(objbst))):
            print('Stop early - 5.00% gap achieved time exceeds 8 minutes')
            model.terminate()
        elif(time>600 and abs(objbst - objbnd) < 0.1 * (1.0 + abs(objbst))):
            print('Stop early - 10.0% gap achieved time exceeds 10 minutes')
            model.terminate()
        elif(time>1500 and abs(objbst - objbnd) < 0.15 * (1.0 + abs(objbst))):
            print('Stop early - 15.0% gap achieved time exceeds 25 minutes')
            model.terminate()
        elif(time>3000 and abs(objbst - objbnd) < 0.2 * (1.0 + abs(objbst))):
            print('Stop early - 20.0% gap achieved time exceeds 50 minutes')
            model.terminate()
        elif(time>6000 and abs(objbst - objbnd) < 0.3 * (1.0 + abs(objbst))):
            print('Stop early - 30.0% gap achieved time exceeds 100 minutes')
            model.terminate()
        elif(time>12000 and abs(objbst - objbnd) < 0.4 * (1.0 + abs(objbst))):
            print('Stop early - 40.0% gap achieved time exceeds 200 minutes')
            model.terminate()
    return


def construct_dummy(synt_net, candidate_edges, evcs):
    graph = synt_net.__class__()
    graph.add_edges_from(synt_net.edges)
    graph.add_edges_from(candidate_edges)

    for e in graph.edges:
        if e in synt_net.edges:
            graph.edges[e]['length'] = synt_net.edges[e]['length']
            graph.edges[e]['r'] = synt_net.edges[e]['r']
        else:
            l = candidate_edges[e] if e in candidate_edges \
                else candidate_edges[(e[1],e[0])]
            graph.edges[e]['length'] = l
            graph.edges[e]['r'] = 0.0822/39690 * l * 1e-3
    for n in graph:
        if n in synt_net:
            graph.nodes[n]['load'] = synt_net.nodes[n]['load']
            graph.nodes[n]['label'] = 'N'
        else:
            graph.nodes[n]['load'] = evcs.demand[n]
            graph.nodes[n]['label'] = 'Y'
    return graph


def get_optimal_routing(synt_net, candidate_edges, evcs, 
                        path, v0=1.0, lambda_=1e3):
    graph = construct_dummy(synt_net, candidate_edges, evcs)
    edgelist = list(graph.edges)
    nodelist = list(graph.nodes)
    n_edges = len(edgelist)
    n_nodes = len(nodelist)
    
    
    R = np.diag([graph.edges[e]['r'] for e in edgelist])
    A = nx.incidence_matrix(graph,nodelist=nodelist,
                            edgelist=edgelist,oriented=True)
    D = nx.incidence_matrix(graph,nodelist=nodelist,
                            edgelist=edgelist,oriented=False)
    
    p = np.array([graph.nodes[n]['load']*1e-3 for n in nodelist])
    l = np.array([graph.edges[e]['length'] for e in edgelist])
    fuel_ind = [i for i,n in enumerate(nodelist) if graph.nodes[n]['label']=='Y']
    
    sublist = [n for n in synt_net if synt_net.nodes[n]['label']=='S']
    root_ind = [nodelist.index(s) for s in sublist]
    pred = np.delete(p,root_ind)
    Ared = np.delete(A.toarray(), root_ind, axis=0)
    
    
    M = 2
    F = sum(pred)
    
    model = grb.Model(name="Get Routing")
    model.ModelSense = grb.GRB.MINIMIZE
    model.write(f"{path}/routing.lp")
    
    x = model.addMVar(
        n_edges, 
        vtype = grb.GRB.BINARY, 
        name = 'x')
    
    f = model.addMVar(
        n_edges,
        vtype=grb.GRB.CONTINUOUS,
        lb=-grb.GRB.INFINITY,
        name='f')
    
    v = model.addMVar(
        n_nodes, 
        vtype=grb.GRB.CONTINUOUS, 
        lb=0.85, ub=1.0, 
        name='v')
    
    w = model.addMVar(
        n_nodes, 
        vtype=grb.GRB.CONTINUOUS, 
        lb=-1, ub=1, 
        name='w')
    
    y = model.addMVar(
        1,
        vtype=grb.GRB.CONTINUOUS, 
        lb=0, ub=1, 
        name='y')
    
    model.addConstr((A.T @ v) - (R @ f) <= M * (1 - x))
    model.addConstr((A.T @ v) - (R @ f) >= M * (x - 1))
    
    for ind in root_ind:
        model.addConstr(v[ind] == v0)
    
    
    model.addConstr((Ared @ f) == -pred)
    model.addConstr(f <= F * x)
    model.addConstr(f >= -F * x)
    
    # Fuel node degree
    model.addConstr( (D[fuel_ind,:] @ x) == 1 )
    
    
    model.addConstr(x.sum() == n_nodes-1)
    for i,e in enumerate(edgelist):
        if e in synt_net.edges:
            model.addConstr( (x[i] == 1) )
    
    # Objective function
    model.addConstr( w == 1.0 - v )
    model.addConstr( y >= (w @ w) )
    model.setObjective( 1e-3 * (l @ x) + (lambda_ * y) )
    model.update()
    
    # Turn off display and heuristics
    grb.setParam('OutputFlag', 0)
    grb.setParam('Heuristics', 0)
    
    # Open log file
    logfile = open(f'{path}/evcs.log', 'w')
    
    # Pass data into my callback function
    # model.params.NonConvex = 2
    model._lastiter = -grb.GRB.INFINITY
    model._lastnode = -grb.GRB.INFINITY
    model._logfile = logfile
    model._vars = model.getVars()
    
    # Solve model and capture solution information
    model.optimize(mycallback)
    # model.optimize()
    
    # Close log file
    logfile.close()
    
    if model.SolCount == 0:
        print(f'No solution found, optimization status = {model.Status}')
        return []
    else:
        x_optimal = x.getAttr("x").tolist()
        optimal_edges = [e for i,e in enumerate(edgelist) if x_optimal[i]>0.8]
        # get new edges
        new_edges = [e for e in optimal_edges \
                     if e in candidate_edges or (e[1],e[0]) in candidate_edges]
        return new_edges



def cvxpy_solve(synt_net, candidate_edges, evcs, 
                v0=1.0, lambda_=1e3, verbose=False):
    
    graph = construct_dummy(synt_net, candidate_edges, evcs)
    edgelist = list(graph.edges)
    nodelist = list(graph.nodes)
    n_edges = len(edgelist)
    n_nodes = len(nodelist)
    
    
    R = np.diag([graph.edges[e]['r'] for e in edgelist])
    A = nx.incidence_matrix(graph,nodelist=nodelist,
                            edgelist=edgelist,oriented=True)
    D = nx.incidence_matrix(graph,nodelist=nodelist,
                            edgelist=edgelist,oriented=False)
    
    p = np.array([graph.nodes[n]['load']*1e-3 for n in nodelist])
    l = np.array([graph.edges[e]['length'] for e in edgelist])
    fuel_ind = [i for i,n in enumerate(nodelist) if graph.nodes[n]['label']=='Y']
    
    sublist = [n for n in synt_net if synt_net.nodes[n]['label']=='S']
    root_ind = [nodelist.index(s) for s in sublist]
    pred = np.delete(p,root_ind)
    Ared = np.delete(A.toarray(), root_ind, axis=0)
    
    
    M = 2
    F = sum(pred)
    
    x = cp.Variable(n_edges, boolean=True)
    f = cp.Variable(n_edges)
    v = cp.Variable(n_nodes)

    constraints = []
    # voltage constraint
    constraints.append((A.T @ v) - (R @ f) <= M * (1 - x))
    constraints.append((A.T @ v) - (R @ f) >= M * (x - 1))
    
    for ind in root_ind:
        constraints.append(v[ind] == v0)
    
    
    constraints.append((Ared @ f) == -pred)
    constraints.append(f <= F * x)
    constraints.append(f >= -F * x)
    
    # Fuel node degree
    constraints.append( (D[fuel_ind,:] @ x) == 1 )
    
    
    constraints.append(cp.sum(x) == n_nodes-1)
    for i,e in enumerate(edgelist):
        if e in synt_net.edges:
            constraints.append( (x[i] == 1) )
    
    # Objective function
    objective = cp.Minimize(1e-3 * (l @ x) + (lambda_ * cp.sum_squares( 1 - v )))
    
    # Solve model and capture solution information
    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=verbose)
    
    # get the solution
    x_optimal = x.value
    optimal_edges = [e for i,e in enumerate(edgelist) if x_optimal[i]>0.8]
    # get new edges
    new_edges = [e for e in optimal_edges \
                    if e in candidate_edges or (e[1],e[0]) in candidate_edges]
    return new_edges
    


























