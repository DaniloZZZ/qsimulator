import qtree
import psutil
import numpy as np

from qensor import utils
from qensor.optimisation.Greedy import GreedyParvars
from loguru import logger as log


class Optimizer:
    def optimize(self, tensor_net):
        raise NotImplementedError

class OrderingOptimizer(Optimizer):
    def _get_ordering_ints(self, graph, free_vars=[]):
        peo_ints, path = utils.get_locale_peo(graph, utils.n_neighbors)

        return peo_ints, path

    def optimize(self, tensor_net):
        line_graph = tensor_net.get_line_graph()
        free_vars = tensor_net.free_vars
        ignored_vars = tensor_net.ket_vars + tensor_net. bra_vars
        graph = line_graph

        if free_vars:
            graph = qtree.graph_model.make_clique_on(graph, free_vars)

        peo, path = self._get_ordering_ints(graph)
        self.treewidth = max(path)

        peo = [qtree.optimizer.Var(var, size=graph.nodes[var]['size'],
                        name=graph.nodes[var]['name'])
                    for var in peo]
        if free_vars:
            peo = qtree.graph_model.get_equivalent_peo(graph, peo, free_vars)

        peo = ignored_vars + peo
        self.peo = peo
        self.graph = graph
        self.ignored_vars = ignored_vars
        return peo, tensor_net

class SlicesOptimizer(OrderingOptimizer):

    def __init__(self, tw_bias=2):
        self.tw_bias = tw_bias

    def _get_max_tw(self):
        mem = psutil.virtual_memory()
        avail = mem.available
        log.info('Memory available: {}', avail)
        # Cost = 16*2**tw
        # tw = log(cost/16) = log(cost) - 4
        return np.int(np.log2(avail)) - 4

    def _split_graph(self, p_graph, max_tw):
        searcher = GreedyParvars(p_graph)
        peo_ints, path = self._get_ordering_ints(p_graph)
        while True:
            #nodes, path = utils.get_neighbours_path(graph, peo=peo_ints)
            tw = self.treewidth
            log.info('Treewidth: {}', tw)
            if tw < max_tw:
                log.info('Found parvars: {}', searcher.result)
                break
            error = searcher.step()
            pv_cnt = len(searcher.result)
            log.debug('Parvars count: {}. Amps count: {}', pv_cnt, 2**pv_cnt)
            if error:
                log.error('Memory is not enough. Max tw: {}', max_tw)
                raise Exception('Estimated OOM')

            peo_ints, path = self._get_ordering_ints(p_graph)
            self.treewidth = max(path)

        return peo_ints, searcher.result

    def optimize(self, tensor_net):
        peo, tensor_net = super().optimize(tensor_net)
        graph = self.graph

        p_graph = graph.copy()
        max_tw = self._get_max_tw()
        log.info('Maximum treewidth: {}', max_tw)
        max_tw = max_tw - self.tw_bias

        peo, par_vars = self._split_graph(p_graph, max_tw)

        # TODO: move these platform-dependent things
        self.parallel_vars = [
            qtree.optimizer.Var(var,
                                size=graph.nodes[var]['size'],
                                name=graph.nodes[var]['name'])
                              for var in par_vars]
        peo = [qtree.optimizer.Var(var, size=graph.nodes[var]['size'],
                        name=graph.nodes[var]['name'])
                    for var in peo]
        if tensor_net.free_vars:
            peo = qtree.graph_model.get_equivalent_peo(p_graph, peo, tensor_net.free_vars)

        self.peo = self.ignored_vars + peo + self.parallel_vars 
        #log.info('peo {}', self.peo)
        return self.peo, self.parallel_vars, tensor_net

class TamakiOptimizer(OrderingOptimizer):
    def __init__(self, *args, wait_time=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.wait_time = wait_time

    def _get_ordering_ints(self, graph):
        peo, tw = qtree.graph_model.peo_calculation.get_upper_bound_peo_pace2017(
                graph, method="tamaki", wait_time=self.wait_time)

        return peo, [tw]

class TreeTrimSplitter(SlicesOptimizer):
    def _split_graph(self, p_graph, max_tw):
        peo_ints, path = self._get_ordering_ints(p_graph)
        graph, _ = utils.reorder_graph(p_graph, peo_ints)
        tw = max(path)
        result = []
        delta = tw - max_tw
        while delta > 0:
            var_target = int((delta + 1)*.8)
            # var_target(1) = 1
            # var_target(2) = 2
            # var_target(15) = 12
            par_vars, p_graph = qtree.graph_model.splitters.split_graph_by_tree_trimming(p_graph, var_target)
            result += par_vars
            peo_ints, path = self._get_ordering_ints(p_graph)
            tw = max(path)
            log.info('Treewidth: {}', tw)
            self.treewidth = tw

            graph, _ = utils.reorder_graph(p_graph, peo_ints)


class TamakiTrimSlicing(TamakiOptimizer, TreeTrimSplitter):
    pass
