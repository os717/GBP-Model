from audioop import avg
import networkx as nx
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy as dc

from .gbp import find_edges
from .utilities import get_running_statistics


def set_plot_options():

    plt.rcParams.update(plt.rcParamsDefault)

    plt.matplotlib.rc('figure', figsize=(9, 5))
    plt.matplotlib.rc('grid', linestyle='dashed', linewidth=1, alpha=0.25)
    plt.matplotlib.rc('font', family='serif', size=12)
    plt.matplotlib.rc('legend', fontsize=12)

    from distutils.spawn import find_executable
    if find_executable('latex'):
        plt.matplotlib.rc('text', usetex=True)

    plt.rcParams['xtick.major.size'] = 7.0
    plt.rcParams['xtick.minor.size'] = 3.0
    plt.rcParams['xtick.direction'] = 'inout'
    plt.rcParams['ytick.major.size'] = 7.0
    plt.rcParams['ytick.minor.size'] = 3.0
    plt.rcParams['ytick.direction'] = 'inout'


def get_plot_colors():
    return ['#6d82ee', '#97964a', '#ffd44f', '#f4777f', '#ad8966']


class NetworkxGraph:
    def __init__(self, A):
        self.A = A
        self.graph = self._initialize_graph()

    def _initialize_graph(self) -> nx.graph:
        """initialize a networkx graph object from data matrix A"""
        graph = nx.Graph()
        for i in range(self.A.shape[0]):
            graph.add_node(i, node_colour='y')

        edges = find_edges(self.A)
        for node in edges:
            graph.add_edge(node[0], node[1], node_colour='y')

        return graph

    def add_new_node(self, num_nodes):
        guardians = [(1, 2), (1, 3)]
        self.graph.add_edges_from(guardians, label="colour")


    def draw_graph(self,filename=None,title=f'', node_size=100):
        # guardians = [(1, 2), (1, 3)]
        """draw a networkx graph"""
        nx.draw(self.graph, node_size=node_size, with_labels=False)
        # nx.draw(self.graph, node_list=guardians, node_colour='y', node_size=node_size)
        # plt.title(title)
        if filename != None:
            str = 'images/' + filename + '.eps'
            plt.savefig(str, format='eps')
            plt.show()

class AnalyzeResult:
    @staticmethod
    def plot_gabp_convergence(iter_dist, color):
        
        # print(iter_dist)
        iter_dist = np.insert(iter_dist, 0, 0)
        # print(iter_dist)

        """plot the distance between each iteration on the GaBP algorithm"""
        fig, ax = plt.subplots()
        plt.semilogy(iter_dist, color=color, label='Total', marker='+')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid()
        ax.set_xlabel('Iteration')
        ax.set_ylabel('L2 Distance')
        ax.set_title('Convergence of Marginals at Every Iteration')
        print(len(iter_dist))
        plt.xlim([1, len(iter_dist)]) 
        plt.show()

    @staticmethod
    def plot_time_vs_iterations(num_iters: int, dims: list):
        colors = get_plot_colors()
        fig, ax = plt.subplots()

        for i, dim in enumerate(dims):
            running_time, num_iter = get_running_statistics(dim, num_iters=num_iters)
            linregress = stats.linregress(running_time, num_iter)

            plt.scatter(running_time, num_iter, color=colors[i], marker='+', label=dim)
            plt.plot(running_time, linregress.intercept + linregress.slope*running_time, color=colors[i], alpha=0.5)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(loc='upper left', title='Number of Nodes', ncol=2)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Iterations till Convergence')
        ax.set_title('Running Time V.S Iterations')
        ax.grid()
        plt.show()

    def gen_deltas(self, x):
        deltas = []
        iters = len(x)
        node_count = len(x[0])

        for i in range(1, iters):
            delta = []
            for j in range(node_count):
                delta.append(abs(x[i][j] - x[i-1][j]))
            deltas.append(dc(delta))
        
        return deltas


    def plot_updates(self, data, metric, **kwargs): 
        show = kwargs.get('show', False)
        convergence_threshold = kwargs.get('convergence_threshold', 1e-5)
        scale = kwargs.get('scale', None)
        name = kwargs.get('name', '---------')
        delta = kwargs.get('delta', False)
        average = kwargs.get('average', False)
        save = kwargs.get('save', False)
        folder = kwargs.get('folder', 'messageupdate')
        
        if delta:
            data = self.gen_deltas(data)

        n = len(data)
        if average:
            plt.plot(range(1, n+1), [np.mean(l) for l in data])
        else:
            for node_data in zip(*data):
                plt.plot(range(1, n+1), node_data)
            # plt.step(range(1, n+1), node_data, where='pre')
        plt.legend(["Node {}".format(i+1) for i in range(len(data[0]))], bbox_to_anchor=(1.005, 1.0), loc='upper left')
        plt.xlabel('Iteration')
        
        start = 0 if metric == 'Random' else 1
        plt.xticks(np.arange(start, n+1, 1))
        plt.xlim([start, n])



        plt.ylabel(metric)
        title = '{}{} of {} using a {} Message Update Schedule'.format('Averaged ' if average else '', 'Evolution' if not delta else 'Convergence', metric, name)
        plt.title(title)
        if show:
            ax = plt.gca()
            ax.axhspan(0, convergence_threshold, alpha=0.2, color='r')
        plt.grid()
        if scale is not None:
            plt.yscale(scale)
        
        plt.tight_layout()
        if save:
            plt.savefig('images/'+folder+'/'+title+'.eps', format='eps')
        plt.show()

    
    def plot_mean_updates(self, data, **kwargs):
        self.plot_updates(data, 'Mean', **kwargs)

    def plot_var_updates(self, data, **kwargs):
        self.plot_updates(data, 'Variance', **kwargs)