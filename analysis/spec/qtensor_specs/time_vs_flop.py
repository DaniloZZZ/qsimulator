# AUTOGENERATED! DO NOT EDIT! File to edit: notebooks/Time_vs_FLOP.ipynb (unless otherwise specified).

__all__ = ['ex', 'graph', 'circuit', 'tn', 'peo', 'sim_costs', 'sum_flops', 'step_flops', 'max_mem', 'SEED',
           'EDGE_IDX_FOR_SEED', 'EDGE_IDX_FOR_SEED_JLSE', 'sim_profile', 'step_sim_time', 'plot_with_filter',
           'get_log_flops_vs_matmul', 'cli', 'time_vs_flops_plot']

# Cell
import sys
import numpy as np
import matplotlib.pyplot as plt

import qtensor as qt
from cartesian_explorer import Explorer

# Cell
import matplotlib as mpl
from cycler import cycler
mpl.rcParams['axes.prop_cycle'] = cycler(color=['#db503d', '#02C6E0'])

# Cell
ex = Explorer()

# Cell
@ex.provider
def graph(n, d, seed):
    return qt.toolbox.random_graph(nodes=n, degree=d, seed=seed)

@ex.provider
def circuit(graph, edge_idx, p):
    gamma, beta = [.1]*p, [.3]*p
    comp = qt.QtreeQAOAComposer(graph, gamma=gamma, beta=beta)
    comp.energy_expectation_lightcone(list(graph.edges())[edge_idx])
    return tuple(comp.circuit)

@ex.provider
def tn(circuit):
    return qt.optimisation.TensorNet.QtreeTensorNet.from_qtree_gates(circuit)

@ex.provider
def peo(tn):
    opt = qt.optimisation.Optimizer.DefaultOptimizer()
    peo, _ = opt.optimize(tn)
    return tuple(peo)

@ex.provider
def sim_costs(tn, peo):
    opt = qt.optimisation.Optimizer.DefaultOptimizer()
    peo, _ = opt.optimize(tn)
    costs, mems = tn.simulation_cost(peo)
    return costs, mems

@ex.provider
def sum_flops(sim_costs):
    flops, mems = sim_costs
    return sum(flops)

# Cell
@ex.provider
def step_flops(sim_costs):
    flops, mems = sim_costs
    return tuple(flops)

@ex.provider
def max_mem(sim_costs):
    flops, mems = sim_costs
    return max(mems)

# Cell
SEED=107

# Cell
EDGE_IDX_FOR_SEED = {
    ,107:  [2, 3, 10, 15]
}

EDGE_IDX_FOR_SEED_JLSE = {
    107:  [2, 4, 8, 14, 15, 21]
}

# Cell
@ex.provider
def sim_profile(circuit, tn):
    backend = qt.PerfNumpyBackend(print=False)
    sim = qt.QtreeSimulator(bucket_backend=backend)

    sim.simulate(circuit)
    data = backend._profile_results
    return tuple(tuple([tuple(x[0]), x[1]]) for x in data.values())

@ex.provider
def step_sim_time(sim_profile, tn):
    ignored_vars = tn.bra_vars+tn.ket_vars
    times = [x[1] for x in sim_profile]
    return tuple(times[len(ignored_vars):])

# Cell
def plot_with_filter(est_flat, times_flat):
    filt = (est_flat>1e4) #& (times_flat>1e-4)
    est_flat_filtered = est_flat[filt]
    times_flat_filtered = times_flat[filt]

    # Fit times
    log_fit_coef = np.polyfit(np.log(est_flat_filtered), np.log(times_flat_filtered), 1)
    fit_coef = np.polyfit(est_flat_filtered, times_flat_filtered, 1)
    print('Lin fit:', fit_coef)
    print('Log fit:', log_fit_coef)
    fit_fn = np.poly1d(log_fit_coef)

    # Plot scatter with filtered data
    plt.scatter(est_flat_filtered, times_flat_filtered)
    xfit = 10**np.linspace(4, 7, 100)
    plt.plot(xfit, np.exp(fit_fn(np.log(xfit))), color='blue')
    plt.loglog()
    plt.xlabel('estimated FLOP')
    plt.ylabel('Runtime')
    return log_fit_coef, fit_coef

# Cell
import timeit
def get_log_flops_vs_matmul(log_fit_coef):
    FLOPS_logfit = np.exp(-log_fit_coef[1])

    N = 300
    matmul_flop = N**2*(N-1)
    x, y = np.random.randn(2, N, N)
    number = 100
    matmul_time = timeit.timeit(lambda: np.matmul(x,y)
                               , number=number)/number

    FLOPS_matmul = matmul_flop/matmul_time

    return FLOPS_logfit, FLOPS_matmul

# Cell
import click

@click.group()
def cli():
    pass

@cli.command()
@click.argument('filename')
def time_vs_flops_plot(filename):
    """
    Plots times and estimated FLOP for each step of several QAOA energy computation contractions.

    Currently using
        - random regular graphs with degree=3,4
        - p = 3
        - N = 1000

    """
    edge_indices = EDGE_IDX_FOR_SEED[SEED]
    ds = [3, 4]
    p = 3
    N = 1000

    estimators = ex.map_variable('step_flops', d=ds, edge_idx=edge_indices, n=[N], p=[p])
    times = ex.map_variable('step_sim_time', d=ds, edge_idx=edge_indices, n=[N], p=[p])

    est_flat = np.concatenate(estimators.flatten())
    times_flat = np.concatenate(times.flatten())

    log_fit_coef, fit_coef = plot_with_filter(est_flat, times_flat)
    plt.savefig(filename)

    fit, matmul = get_log_flops_vs_matmul(log_fit_coef)

    print('===Results===')
    print(f'Simulator fitted flops: {fit/1e9:.5} G')
    print(f'Matmul flops: {matmul/1e9:.5} G')
    print(f'Simulator optimality: {fit/matmul}')