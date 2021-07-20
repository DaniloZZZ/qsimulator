from numpy.core.fromnumeric import size
import pyrofiler
from typing import List
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
import itertools
import platform
import importlib

from base import LasyModule, BenchResult, Backend, get_gpu_props_json, Benchmark, Numpy, TorchCuda, Cupy, CuTensor

np = LasyModule('numpy')
torch = LasyModule('torch')
cupy = LasyModule('cupy')
cupy_cutensor = LasyModule('cutensor')
import sys
import os
sys.path.append(os.environ['HOME']+"/.local")
exatn = LasyModule('exatn')


class MatmulBench(Benchmark):
    @classmethod
    def __init__(self, backend:Backend):
        self.backend = backend
    
    @staticmethod
    def get_task_type():
        return "matmul"

    @classmethod
    def get_operation(cls):
        return cls.backend.get_matmul()
    
    @classmethod
    def get_params(cls, *sizes):
        ops = np.prod(sizes[0]) * sizes[1][1]
        param_in = np.prod(sizes[0]) + np.prod(sizes[1])
        param_out = sizes[0][0]*sizes[1][1]
        return ops.item(), param_in.item(), param_out


class CuTensorMatmul(CuTensor):
    @classmethod
    def get_ready(self, num_tensors, *sizes):
        sizes = list(sizes)
        num_tensors += 1
        sizes.append([sizes[0][0], sizes[1][1]])
        return num_tensors, *sizes
    

# @dataclass
# class ExatnTensor:
#     name: str
#     shape: tuple
#     dtype: str

# class Exatn(Backend):
#     # infinite name generator
#     name_generator = (hex(x)[1:] for x, _
#                       in enumerate(itertools.repeat(0)))
#     allocated_tensor_names = []
#     @classmethod
#     def cleanup_tensors(cls):
#         for name in cls.allocated_tensor_names:
#             # will produce a warning on non-existent tensor
#             exatn.destroyTensor(name)
#         cls.allocated_tensor_names = []

#     @staticmethod
#     def get_dtype(dtype):
#         return Numpy.get_dtype(dtype)

#     @classmethod
#     def gen_tensor(cls, *sizes, dtype='float'):
#         tname = next(cls.name_generator)
#         rand = Numpy.gen_tensor(*sizes, dtype=dtype)
#         #print('create ten', tname)
#         success = exatn.createTensor(tname, rand.copy(order='F'))
#         #print('done', tname)
#         if success:
#             cls.allocated_tensor_names.append(tname)
#             return ExatnTensor(name=tname, shape=sizes, dtype=dtype)

#     @classmethod
#     def exatn_matmul(cls, x, y):
#         """
#         Takes two names of tensors, should be already allocated,
#         returns name of resulting tensor

#         Args:
#             x: ExatnTensor
#             y: ExatnTensor
#         """
#         #exatn.evaluateTensorNetwork('sum', 'SR1() = R1(a)*R2(a)')
#         res = next(cls.name_generator)
#         res_shape = x.shape[0], y.shape[1]
#         #print('create res', res, res_shape)
#         dtype = Numpy.get_dtype(x.dtype)
#         res_body = np.zeros(res_shape, dtype=dtype)
#         exatn.createTensor(res, res_body)
#         cls.allocated_tensor_names.append(res)
#         st = f'{res}(a,c)={x.name}(a,b)*{y.name}(b,c)'
#         #print('st', st)
#         _ = exatn.contractTensors(st, 1.0)
#         #print('contr')
#         return res

#     @classmethod
#     def get_operation(cls):
#         return cls.exatn_matmul

#     @classmethod
#     def get_result(cls, x):
#         t = exatn.getLocalTensor(x)
#         cls.cleanup_tensors()
#         return t





def main():

    experiment_group = "Angela_nslb_matmul_test"


    num_tensors = 2
    sizes_m = [10, 100, 1000, 1024, 1025, 2000, 4090, 4096, 8192]
    sizes_n = [10, 100, 1000, 1024, 1025, 2000, 4090, 4096, 8192]
    sizes_l = [10, 100, 1000, 1024, 1025, 2000, 4090, 4096, 8192]
    # sizes_m = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

    

    backends = {
        'numpy':Numpy
        # ,'exatn': Exatn
    }
    if get_gpu_props_json():
        backends.update({
            'torch':TorchCuda
            , 'cupy':Cupy
            , 'cutensor': CuTensorMatmul
        })

    use_strip = True
    repeats = 5
    if use_strip:
        repeats += 2
    dtypes = ['float', 'double', 'complex64', 'complex128']

    for backend in backends:
        for size_m, size_n, size_l in zip(sizes_m, sizes_n, sizes_l):
            sizes = [size_m,size_n], [size_n, size_l]
            results = []
            for dtype in dtypes:
                for _ in range(repeats):
                    b = backends[backend]
                    matmulbench = MatmulBench(b)
                    _, bench_result = matmulbench.benchmark(b, num_tensors, *sizes, dtype=dtype)
                    results.append(bench_result)
                json_result = matmulbench.print_results_json(use_strip, backend, *sizes, dtype=dtype, results=results, experiment_group=experiment_group)

if __name__ == "__main__":
    main()

