import json
from functools import lru_cache
import numpy as np
import networkx as nx
import pytest
import platform
import pyrofiler
from qtensor import QtreeQAOAComposer
from qtensor import QAOAQtreeSimulator
from qtensor.contraction_backends import get_backend, get_perf_backend

'''
# Edit 1:
# 7/8/2021
# Notes:
# test_qaoa_energy_multithread is the function calling for testing
# The current objective is to make the simulators run on different backends
# NOW: simulators are running on vanilla
# NEED TO DO:
    1. Import cupybackend
    2. Feed it to the simulation

# Edit 2:
# 7/8/2021:
# Above Edit 1 is solved supraphysiologically
# Current Issue:
#   1. Useful Benchmarking result in printed not captured: sim.backend.gen_report()
#   2. Imcompatible backends exist
# Objective:
#   1. Mimic Angela's benchmarking approach:
#       a. Create a generic backend
#   2. Mimic Dan's circuit.py
# Proposal:
#   1. Integrate Timing Components in the function, so we do not need to write a seperate class
#   2. Using bench.py's wrapper class

Edit 3:
7/10/2021:
1. Trying to pass the decorated backends into the benchmarking code


Edit 4:
7/11/2021:
Info Obtained, Need to tidy
1. Current Design:
    a. Iterate 7 time, each iter generates a mean report, will be 7 mean report for each backend
        i. result_dict[backend] = full_report, full_report[field_name] = array[7] of data


Edit 5:
7/12/2021
1. Need to fixate PEO (simulator.py)
2. Refer to Angele's json format
3. Ask Angela for MongoDB use
'''


# TODO: Make a larger graph for benchmarking: Increase p 
@lru_cache
def get_test_problem(n=10, p=2, d=3, type='random'):
    print('Test problem: n, p, d', n, p, d)
    if type == 'random':
        G = nx.random_regular_graph(d, n)
    elif type == 'grid2d':
        G = nx.grid_2d_graph(n,n)
    elif type == 'line':
        G = nx.Graph()
        G.add_edges_from(zip(range(n-1), range(1, n)))
    gamma, beta = [np.pi/5]*p, [np.pi/2]*p
    return G, gamma, beta

@pytest.fixture
def test_problem(request):
    n, p, d, type = request.param
    return get_test_problem(n, p, d, type)


paramtest = [
    # n, p, degree, type
     [4, 4, 3, 'random']
    ,[10, 5, 2, 'random']
    ,[14, 1, 3, 'random']
    ,[3, 3, 0, 'grid2d']
    ,[8, 4, 0, 'line']
]

def mean_mmax(x: list):
    mx, mn = max(x), min(x)
    x.remove(mx)
    x.remove(mn)
    return np.mean(x)

def format_flops(flops):
    ord = 3*int(np.log10(flops)/3)
    suffix = {
        3: 'k'
        ,6: 'M'
        ,9: 'G'
        , 12: 'T'
    }[ord]
    return f'{(flops/10**ord).round(2)}{suffix}'

def obj2dict(obj):
    keys = [x for x in dir(obj) if x[0]!='_']
    return dict((key, obj.__getattribute__(key)) for key in keys)


@lru_cache
def get_gpu_props_json():
    try:
        import torch
        devprops = torch.cuda.get_device_properties(torch.cuda.current_device())
        return obj2dict(devprops)
    except:
        return None



'''
My Edit:
'''

'''
NEED to 
1. gen_time is not important, but needs to check regardless
    a. Try 
2. bytes = number of initial tensor, probly in perf_backend: the total memory size
3. mult_time= summmation over time per repeat, mult_relstd= std of the 5 time tidies
4. op not flops (bascically flop)
5. width is the max_size
6. problem_def for the initial
    a. random seed can be obtained by qtensor
'''
def test_qaoa_energy_vs_backends(backend_name: str, show = False):

    # 1. Preperation Phase
    timing = pyrofiler.timing
    with timing(callback=lambda x: None) as gen:
        G, gamma, beta = get_test_problem(10, 3, 3)
        curr_backend = get_perf_backend(backend_name)
        sim = QAOAQtreeSimulator(QtreeQAOAComposer, backend = curr_backend)
    res = sim.energy_expectation(G, gamma=gamma, beta=beta)

    # 2. Data Collection Phase
    curr_backend.gen_report(show = show)
    #print(list(curr_backend._profile_results.keys()))

    # For each bucket_sign, it has a list of tensor_sign
    # Each Tensor_sign is a list of qtree.var
    #
    tensor_dims= []
    for x in curr_backend._profile_results:
        bucket_signature, _ = curr_backend._profile_results[x]
        for tensor in bucket_signature:
            tensor_2_size = [qtreeVar.size for qtreeVar in tensor]
            tensor_dim = np.prod(tensor_2_size)
            tensor_dims.append(tensor_dim)
            # print()
            # print(tensor)
            # print(tensor_2_size)
            # print(tensor_dim)

    
    return curr_backend.report_table , np.sum(tensor_dims), gen.result, len(curr_backend.report_table.records)




# Return a dictionary of: backend -> report
# where report is a dict: benchmark attributes -> array of means of each iteration
def qaoa_energy_benchmarking(repeat = 7):
    backend_names = ["cupy", "einsum", "torch", 'tr_einsum','opt_einsum']
    test_names = ["cupy"]
    #test_name = ["einsum"]
    backend_2_report = {}
    error_list = []
    titles = None


    # 1. Generating Raw Data
    for name in backend_names:
        means_frame = []
        sums_frame = []
        maxs_frame = []
        byte_frame = []
        gen_time_frame = []
        bucket_len_frame = []
        title_2_means = {}
        try:
            for i in range(repeat):
                iter_report, byte, gen_time, num_buckets= test_qaoa_energy_vs_backends(name)
                byte_frame.append(byte)
                gen_time_frame.append(gen_time)
                bucket_len_frame.append(num_buckets)
                titles = iter_report._title_row()[1:]
                means = np.mean(iter_report.records, axis = 0)
                means_frame.append(means)
                sums = np.sum(iter_report.records, axis = 0)
                sums_frame.append(sums)
                maxs = np.amax(iter_report.records, axis = 0)
                maxs_frame.append(maxs)
        except Exception as e:
            error_list.append(name)
            print(e)
        
        means_frame = np.array(means_frame)
        means_frame = np.transpose(means_frame)
        for i,title in enumerate(titles):
            title_2_means["mean_"+title] = list(means_frame[i])

        sums_frame = np.array(sums_frame)
        sums_frame = np.transpose(sums_frame)
        for i,title in enumerate(titles):
            title_2_means["sum_"+title] = list(sums_frame[i])
        
        maxs_frame = np.array(maxs_frame)
        maxs_frame = np.transpose(maxs_frame)
        for i,title in enumerate(titles):
            title_2_means["max_"+title] = list(maxs_frame[i])

        title_2_means["byte"] = byte_frame
        title_2_means["gen_time"] = gen_time_frame
        title_2_means["bucket_len"] = bucket_len_frame




        backend_2_report[name] = title_2_means
    #print(backend_2_report)
    return backend_2_report




# Raw_Reports are backend->report 
# where report is: col_name/attribute -> array of means
# Return backend -> (report -> caliborated mean)
def cook_raw_report(raw_reports, task_type = "QAOAEnergyExpectation"):

    # 1. Condense the repetition
    medium_reports = {}
    for be, report in raw_reports.items():
        col_2_mean = {}
        for col, arr in report.items():
            new_datum = mean_mmax(arr) #
            col_2_mean[col] = new_datum
        medium_reports[be] = col_2_mean
    
    # 2. Produce final report
    cooked_reports = []
    GPU_PROPS = get_gpu_props_json()
    for backend, reports in medium_reports.items():
        res = dict(
            backend = backend,
            flops = reports['mean_FLOPS'],
            flops_str = format_flops(reports['mean_FLOPS']),
            device_props=dict(name=platform.node(), gpu=GPU_PROPS),
            task_type = task_type,
            ops = reports["sum_flop"],
            width = reports["max_max_size"], 
            mult_time = reports["sum_time"],
            mult_relstd = np.std(raw_reports[backend]["sum_time"]),
            bytes = reports["byte"],
            gen_time = reports["gen_time"],
            num_buckets = reports["bucket_len"]
        )
        cooked_reports.append(res)
    return cooked_reports


# 3. check backend tr_einsum for fault
if __name__ == '__main__':
    
    # Should I use any or all?
    # raw_reports = qaoa_energy_benchmarking(repeat = 7)
    # cooked_reports = cook_raw_report(raw_reports)
    # with open("sample.json", "w") as outfile:
    #     json.dump(cooked_reports, outfile, indent = 4, sort_keys= True)
    
    G, gamma, beta = get_test_problem(10, 50, 3, type = "random")
    curr_backend = get_backend("cupy")
    sim = QAOAQtreeSimulator(QtreeQAOAComposer, backend = curr_backend)
    res = sim.energy_expectation(G, gamma=gamma, beta=beta)