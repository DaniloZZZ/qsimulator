from numpy.core.fromnumeric import size
import pyrofiler
from typing import List
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
import itertools
import platform
import importlib

class LasyModule:
    def __init__(self, modulename):
        self.modulename = modulename
        self.module = None
    def __getattr__(self, attr):
        if self.module is None:
            self.module = importlib.import_module(self.modulename)
        return self.module.__getattribute__(attr)

np = LasyModule('numpy')
torch = LasyModule('torch')
cupy = LasyModule('cupy')
cupy_cutensor = LasyModule('cutensor')
import sys
import os
sys.path.append(os.environ['HOME']+"/.local")
exatn = LasyModule('exatn')
from cupy import cutensor as cupy_cutensor

@dataclass
class BenchResult:
    gen_time: float
    transfer_time: float
    operation_time: float


class Backend:
    @classmethod
    def gen_tensor(cls, *sizes, dtype='float'):
        raise NotImplementedError
    
    @classmethod
    def get_ready(cls, num_tensors, *sizes):
        return num_tensors, *sizes

    @staticmethod
    def prepare(x):
        return x

    @staticmethod
    def get_result(x):
        return x
    
    @classmethod
    def gen_tensors(cls, num_tensors, *sizes, dtype='float'):
        tensors = []
        assert num_tensors == len(sizes), "number of tensors does not match input size array"
        for i in range(num_tensors):
            tensor = cls.gen_tensor(*sizes[i], dtype=dtype)
            tensors.append(tensor)
        return tensors
    


class Benchmark:
    timing=pyrofiler.timing

    @staticmethod
    def get_task_type():
        raise NotImplementedError

    @classmethod
    def get_operation(cls):
        raise NotImplementedError
        
    @classmethod
    def benchmark(cls, backend:Backend, num_tensors, *sizes, dtype='float', **args):
        num_tensors, *sizes = backend.get_ready(num_tensors, *sizes)
        operation = cls.get_operation()
        with cls.timing(callback=lambda x: None) as gen:
            tensors = backend.gen_tensors(num_tensors, *sizes, dtype=dtype)
        with cls.timing(callback=lambda x: None) as prep:
            for i in range(len(tensors)):
                tensors[i] = backend.prepare(tensors[i])
        with cls.timing(callback=lambda x: None) as op:
            if 'contraction' in args:
                out_tensor = operation(args['contraction'], *tensors)
            else:
                out_tensor = operation(*tensors)
        with cls.timing(callback=lambda x: None) as get:
            zr = backend.get_result(out_tensor)
        return zr, BenchResult(gen_time=gen.result, transfer_time=prep.result+get.result, operation_time=op.result)



    def format_flops(flops):
        ord = 3*int(np.log10(flops)/3)
        suffix = {
            3: 'k'
            ,6: 'M'
            ,9: 'G'
            , 12: 'T'
            , 15: 'P'
        }[ord]
        return f'{(flops/10**ord).round(2)}{suffix}'

    def get_dtype_size(dtype):
        dtype_t = {
            'float':np.float32
            ,'double': np.float64
            ,'complex64': np.complex64
            ,'complex128': np.complex128
        }[dtype]
        x = np.ones(10, dtype=dtype_t)
        return x.itemsize

    def mean_mmax(x: list):
        mx, mn = max(x), min(x)
        x.remove(mx)
        x.remove(mn)
        return np.mean(x)


    @classmethod
    def get_params(cls, **args):
        raise NotImplementedError


    @classmethod
    def print_results_json(cls, use_strip, backend, *sizes, dtype, results: List[BenchResult], experiment_group="default group"):
        import json
        GPU_PROPS = get_gpu_props_json()
        tt1 = [r.gen_time for r in results]
        tt2 = [r.operation_time for r in results]
        tt3 = [r.transfer_time for r in results]
        m1, m3 = np.mean(tt1), np.mean(tt3)
        if use_strip:
            m1 = cls.mean_mmax(tt1)
            m2 = cls.mean_mmax(tt2)
            m3 = cls.mean_mmax(tt3)
        else:
            m1, m2, m3 = np.mean(tt1), np.mean(tt2), np.mean(tt3)
        s1, s2, s3 = np.std(tt1), np.std(tt2), np.std(tt3)
        ops, param_in, param_out = cls.get_params(*sizes)
        flops = ops/m2
        task_type = cls.get_task_type()
        res = dict(
            task_type=task_type
            , backend=backend
            , size=sizes[0][0]
            , sizes=sizes
            , itemsize=cls.get_dtype_size(dtype)
            , input_bytes=cls.get_dtype_size(dtype)*param_in
            , output_bytes=cls.get_dtype_size(dtype)*param_out
            , dtype=dtype
            , device_props=dict(name=platform.node(), gpu=GPU_PROPS)
            , transfer_time=m3
            , transfer_relstd=s3
            , gen_time=m1
            , gen_relstd=s1
            , operation_time=m2
            , operation_relstd=s2
            , ops=ops
            , flops=flops
            , flops_str=cls.format_flops(flops)
            , fbratio=ops/(cls.get_dtype_size(dtype)*param_in)
            , experiment_group=experiment_group
        )
        print(json.dumps(res), flush=True)
        return res


class Numpy(Backend):
    @staticmethod
    def get_dtype(dtype):
        return {
            'float':np.float32
            ,'double': np.float64
            ,'complex64': np.complex64
            ,'complex128': np.complex128
        }[dtype]

    @classmethod
    def gen_tensor(cls, *sizes, dtype='float'):
        dtype_t = cls.get_dtype(dtype)
        return np.random.rand(*sizes).astype(dtype_t)

    @staticmethod
    def get_matmul():
        return np.matmul
    
    @staticmethod
    def get_tncontract():
        return np.einsum


class Torch(Backend):
    torch.backends.cuda.matmul.allow_tf32 = False
    gpu_tensor = ['float', 'double']

    @staticmethod
    def get_dtype(dtype):
        return {
            'float':torch.cuda.FloatTensor
            ,'double': torch.cuda.DoubleTensor
            ,'complex64': torch.complex64
            ,'complex128': torch.complex128
        }[dtype]

    @classmethod
    def gen_tensor(cls, *sizes, dtype='float'):
        if dtype in cls.gpu_tensor:
            dtype = cls.get_dtype(dtype)
            return dtype(*sizes).normal_()
        else:
            dtype = cls.get_dtype(dtype)
            return torch.rand(*sizes, dtype=dtype, device='cuda')

    @staticmethod
    def get_matmul():
        return torch.matmul
    
    @staticmethod
    def get_tncontract():
        return torch.einsum


class TorchCuda(Torch):
    @classmethod
    @contextmanager
    def timing(cls, **kwargs):
        class Foo:
            pass
        res = Foo()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        yield res
        end.record()
        torch.cuda.synchronize()
        res.result = start.elapsed_time(end)/1000


class Cupy(Backend):
    @classmethod
    @contextmanager
    def timing(cls, **kwargs):
        class Foo:
            pass
        res = Foo()
        start = cupy.cuda.Event(disable_timing=False)
        end = cupy.cuda.Event(disable_timing=False)
        start.record()
        yield res
        end.record()
        end.synchronize()
        res.result = cupy.cuda.get_elapsed_time(start, end)/1000

    @staticmethod
    def get_dtype(dtype):
        return {
            'float':cupy.float32
            ,'double': cupy.float64
            ,'complex64': cupy.complex64
            ,'complex128': cupy.complex128
        }[dtype]

    @classmethod
    def gen_tensor(cls, *sizes, dtype='float'):
        dtype_t = cls.get_dtype(dtype)
        return cupy.random.rand(*sizes).astype(dtype_t)

    @staticmethod
    def get_matmul():
        with pyrofiler.timing('cblas handler'):
            cupy.cuda.device.get_cublas_handle()
        return cupy.matmul

    @staticmethod
    def get_tncontract():
        with pyrofiler.timing('cblas handler'):
            cupy.cuda.device.get_cublas_handle()
        return cupy.einsum


class CuTensor(Cupy):
    @classmethod
    def get_ready(self, num_tensors, *sizes):
        raise NotImplementedError

    @classmethod
    def gen_tensor(cls, *sizes, dtype='float'):
        dtype_t = cls.get_dtype(dtype)
        return cupy.random.rand(*sizes).astype(dtype_t)

    @classmethod
    def cutensor_matmul(cls, x, y, z):
        [x, desc_x] = x
        [y, desc_y] = y
        [z, desc_z] = z
        from cupy import cutensor
        return cutensor.contraction(1, x, desc_x,  ('m', 'k'), 
                                y, desc_y, ('k', 'n'), 0, 
                                z, desc_z, ('m', 'n'))
    
    @classmethod
    def get_matmul(cls):
        with pyrofiler.timing('cblas handler'):
            cupy.cuda.device.get_cublas_handle()
        return cls.cutensor_matmul
    
    @classmethod
    def prepare(cls, x):
        desc_x = cupy_cutensor.create_tensor_descriptor(x)
        return [x, desc_x]
    
    @classmethod
    def tncontract(cls, contraction, *tensors):
        [x, desc_x] = tensors[0]
        [y, desc_y] = tensors[1]
        [z, desc_z] = tensors[2]
        return cupy_cutensor.contraction(1.0, x, desc_x, ('A', 'B', 'C', 'D'), 
                        y, desc_y, ('B', 'C', 'D', 'F'), 0, 
                        z, desc_z, ('A', 'C', 'F'))

    @classmethod
    def get_tncontract(cls):
        with pyrofiler.timing('cblas handler'):
            cupy.cuda.device.get_cublas_handle()
        return cls.tncontract



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