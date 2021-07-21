#from torch._C import device
from .base_class import ContractionBackend
from .numpy import NumpyBackend
from .torch import TorchBackend
from .cupy import CuPyBackend
from .mkl import CMKLExtendedBackend
from .transposed import TransposedBackend
from .opt_einsum import OptEinusmBackend
from .performance_measurement_decorator import PerfNumpyBackend, PerfBackend

def get_backend(name):
    backend_dict = {
        'einsum':NumpyBackend,
        'torch': TorchBackend,
        'mkl':CMKLExtendedBackend,
        'tr_einsum': TransposedBackend,
        'opt_einsum': OptEinusmBackend,
        'cupy': CuPyBackend
    }
    if name in ["torch_gpu", "torch_cpu"]:
        return backend_dict['torch'](device = name[-3:])
    else:
        return backend_dict[name]()

def get_perf_backend(name):
    class MyPerfBackend(PerfBackend):
        Backend = {
            'einsum':NumpyBackend,
            'torch': TorchBackend,
            'torch_cpu': TorchBackend,
            'torch_gpu': TorchBackend,
            'mkl':CMKLExtendedBackend,
            'tr_einsum': TransposedBackend,
            'opt_einsum': OptEinusmBackend,
            'cupy': CuPyBackend,
        }[name]

    if name in ["torch_gpu", "torch_cpu"]:
        return MyPerfBackend(device = name[-3:])
    else:
        return MyPerfBackend()
