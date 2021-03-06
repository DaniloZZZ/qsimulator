import time

import numpy as np
import exatn as tn

def benchmark(num_iter, num_batch, f, *args):
    acc = 0;
    for idx in range(num_iter):
        start = time.time()
        for jdx in range(num_batch):
            f(*args)
        acc += (time.time() - start) / num_batch
    return acc / num_iter

def foo():
    tn.evaluateTensorNetwork("Bench", "C(i, k) = A(j, i) * B(j, k)")


def run(n, num_iter, num_batch):
    tn.createTensor("C", np.zeros((n, n)))
    tn.createTensor("A", np.random.rand(n, n))
    tn.createTensor("B", np.random.rand(n, n))
    nops = 2 * n ** 3 / 1e9
    elapsed = benchmark(num_iter, num_batch, foo)
    tn.getLocalTensor("C")
    tn.destroyTensor("A")
    tn.destroyTensor("B")
    tn.destroyTensor("C")  
    print(n, ", ", nops / elapsed)

if __name__ == "__main__":
    for i in range(4102, 4090 - 1, -1):
        run(i, 20, 1)

