import json
from functools import lru_cache
import numpy as np
import networkx as nx
import platform
import pyrofiler
from qtensor import QtreeQAOAComposer
from qtensor import QAOAQtreeSimulator
from qtensor import toolbox
from qtensor.contraction_backends import get_backend, get_perf_backend

@lru_cache
def get_test_problem(n=10, p=2, d=3, type='random'):
    print('Test problem: n, p, d', n, p, d)
    if type == 'random':
        seed = 250
        G = nx.random_regular_graph(d, n, seed = 250)
    elif type == 'grid2d':
        G = nx.grid_2d_graph(n,n)
    elif type == 'line':
        G = nx.Graph()
        G.add_edges_from(zip(range(n-1), range(1, n)))
    gamma, beta = [np.pi/5]*p, [np.pi/2]*p
    return G, gamma, beta

paramtest = [
    # n, p, degree, type
     [4, 4, 3, 'random']
    ,[10, 5, 2, 'random']
    ,[14, 1, 3, 'random']
    ,[3, 3, 0, 'grid2d']
    ,[8, 4, 0, 'line']
]

paramtest_p = [
    [10, 1, 3, 'random'],
    [20, 2, 3, 'random'],
    [30, 3, 3, 'random'],
    [40, 4, 3, 'random'],
    [10, 5, 3, 'random'],
]

def param_gen(p_max, n_max, type:str):
    result = []
    for n in range(24, n_max+10, 10):
        result.append([n,p_max, 3, type])
    return result   









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
Main Functions:
'''

'''
Function: Generate a single report of a backend running a problem
I/O: backend_name, problem_descrp -> report table, byte, gen_time, num_buckets
'''
def gen_be_pt_report(backend_name:str , pt: list):
    n, p, d, type = pt
    timing = pyrofiler.timing
    with timing(callback=lambda x: None) as gen:
        G, gamma, beta = get_test_problem(n=n,p=p,d=d, type = type)
        curr_backend = get_perf_backend(backend_name)
        opt = toolbox.get_ordering_algo('rgreedy_0.05_30')
        sim = QAOAQtreeSimulator(QtreeQAOAComposer, backend = curr_backend, optimizer = opt)
    sim.energy_expectation(G, gamma=gamma, beta=beta)
    curr_backend.gen_report(show = False)

    tensor_dims= []
    for x in curr_backend._profile_results:
        bucket_signature, _ = curr_backend._profile_results[x]
        for tensor in bucket_signature:
            tensor_2_size = [qtreeVar.size for qtreeVar in tensor]
            tensor_dim = np.prod(tensor_2_size)
            tensor_dims.append(tensor_dim)

    report_record = curr_backend.report_table.records

    mean_record = np.mean(report_record, axis = 0)
    max_record = np.amax(report_record, axis = 0)
    sum_record = np.sum(report_record, axis = 0)
    title_record = curr_backend.report_table._title_row()[1:]
    byte = np.sum(tensor_dims)
    gen_time = gen.result
    num_buckets = len(curr_backend.report_table.records)

    return max_record, mean_record, sum_record, title_record, byte, gen_time, num_buckets


'''
Function: Generate a collection of above report, and process them into final usable form
I/O: ... -> processed data is a dict, directly usable by json
'''
def collect_process_be_pt_report(repeat: int, backend_name: str, pt: list):
    means_frame = []
    sums_frame = []
    maxs_frame = []
    byte_frame = []
    gen_time_frame = []
    bucket_num_frame = []
    title_2_data = {}

    for _ in range(repeat):
        maxs, means, sums, titles, byte, gen_time, num_buckets = gen_be_pt_report(backend_name = backend_name, pt = pt)
        maxs_frame.append(maxs)
        means_frame.append(means)
        sums_frame.append(sums)
        byte_frame.append(byte)
        gen_time_frame.append(gen_time)
        bucket_num_frame.append(num_buckets)

    means_frame = np.array(means_frame)
    means_frame = np.transpose(means_frame)
    sums_frame = np.array(sums_frame)
    sums_frame = np.transpose(sums_frame)
    maxs_frame = np.array(maxs_frame)
    maxs_frame = np.transpose(maxs_frame)
    for i, title in enumerate(titles):
        title_2_data["mean_"+title] = list(means_frame[i])
        title_2_data["sum_"+title] = list(sums_frame[i])
        title_2_data["max_"+title] = list(maxs_frame[i])
    
    title_2_data["bytes"] = byte_frame
    title_2_data["gen_time"] = gen_time_frame
    title_2_data["bucket_num"] = bucket_num_frame
    
    return title_2_data





def cook_raw_report(backend_name: str, problem: list, raw_report: dict, task_type = "QAOAEnergyExpectation"):
    
    medium_report = {}
    for title, arr in raw_report.items():
        new_mean = mean_mmax(arr)
        medium_report[title] = new_mean

    GPU_PROPS = get_gpu_props_json()
    res = dict(
        backend = backend_name,
        device_props = dict(name=platform.node(), gpu=GPU_PROPS),
        task_type = task_type,
        flops = medium_report["mean_FLOPS"],
        flops_str = format_flops(medium_report["mean_FLOPS"]),
        ops = medium_report["sum_flop"],
        width = medium_report["max_max_size"],
        mult_time = medium_report["sum_time"],
        mult_relstd = np.std(raw_report["sum_time"]),
        bytes = medium_report["bytes"],
        gen_time = medium_report["gen_time"],
        num_buckets = medium_report["bucket_num"],
        problem = {
                    "n" :problem[0] , 
                    "p" :problem[1] ,
                    "d" :problem[2] ,
                    'type': problem[3]
                    }
    )

    return res

if __name__ == '__main__':


    total_report = []
    backends= ["cupy","einsum", "torch","torch_gpu", 'tr_einsum','opt_einsum']
    problems = param_gen(4,100,"random")
    for be in ["torch"]:
        for pt in [problems[0]]:
            raw_report = collect_process_be_pt_report(7, be, pt)
            cooked = cook_raw_report(be, pt, raw_report)
            total_report.append(cooked)
            print(json.dumps(cooked))
    with open("a100.json","w") as outfile:
        json.dump(total_report, outfile, indent=4)
    print(total_report)
    

    # G, gamma, beta = get_test_problem(4,4,3, type = "random")
    # curr_backend = get_perf_backend("torch")
    # print(curr_backend.backend.device)
    # print(curr_backend.backend.cuda_available)
    # sim = QAOAQtreeSimulator(QtreeQAOAComposer, backend = curr_backend)
    # print(sim.energy_expectation(G, gamma=gamma, beta=beta))




# TODO:
# 1. increase n, increase complexity
# 2. n = 30, p = 5, d = 3, "random", p = 5 is the hard cap for p
#
