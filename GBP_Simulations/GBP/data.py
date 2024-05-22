import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import networkx as nx
import matplotlib as plt

from .gbp import find_edges

# class DataTest:
#     def symmetric(a: np.array) -> np.array:
#         """Return a symmetrical version of NumPy array a."""
#         return a + a.T - np.diag(np.diag(a))

#     def get_random_data(self, dim, sparcity_threshold=-0.25):
#         """Return a sparse, symmetric, positive, normal distributed and normalized data matrix and observations vector"""
#         A_rand = np.random.randn(dim, dim)
#         print(A_rand < sparcity_threshold)
#         A_sparce = A_rand * (A_rand < sparcity_threshold)

#         A = self.symmetric(np.triu(A_sparce)) + np.diag(np.diag(A_rand))
#         A = np.abs(A)
#         A = A / np.sum(A)

#         b = np.abs(np.random.randn(dim, 1))
#         b = b / np.sum(b)
#         b = b.reshape(-1)

#         return A, b

#     def get_sparse_tree_matrix(self, dim, sparcity_threshold=-0.25):
#         """Use minimum spanning tree to convert a sparse matrix for a sparse tree matrix"""
#         A, b = self.get_random_data(dim, sparcity_threshold)
#         A_diag = np.diag(np.diag(A))
#         A_up_triangle = np.triu(A) - A_diag
#         A_sparse = csr_matrix(A_up_triangle)
#         A_min_span_tree = minimum_spanning_tree(A_sparse).toarray()
#         A = A_min_span_tree + A_diag
#         A = self.symmetric(A)
#         return A, b

class DataGenerator:
    @staticmethod
    def symmetric(a: np.array) -> np.array:
        """Return a symmetrical version of NumPy array a."""
        return a + a.T - np.diag(np.diag(a))

    def get_test_data(self, dim):
        """Return a symmetric, positive, normally distributed, and normalized data matrix and observation vector"""
        A = np.random.randn(dim, dim)
        A = (A + A.T) / 2  # Ensure symmetry
        A = np.abs(A)  # Ensure positivity
        A /= np.sum(A)  # Normalize

        b = np.random.randn(dim, 1)
        b = np.abs(b)  # Ensure positivity
        b /= np.sum(b)  # Normalize
        b = b.reshape(-1)

        return A, b
    
    def get_test_data(self, dim, mean_connectivity, connectivity_std=1.0):
        """Return a symmetric, positive, normally distributed, and normalized data matrix and observation vector
        with the number of nodes each node is connected to described by a normal distribution."""
        # Generate a random symmetric matrix
        A = np.random.randn(dim, dim)
        A = (A + A.T) / 2  # Ensure symmetry
        A = np.abs(A)  # Ensure positivity
        
        # Generate random number of connections for each node based on normal distribution
        num_connections = np.random.normal(mean_connectivity, connectivity_std, dim)
        num_connections = np.maximum(num_connections, 0)  # Ensure non-negative values

        # Create a diagonal matrix with the number of connections
        D = np.diag(num_connections)

        # Generate the adjacency matrix with random connections
        A = np.random.randn(dim, dim)
        A = np.sign(A) * (np.random.rand(dim, dim) < num_connections / dim)

        # Make sure the matrix is symmetric
        A = np.maximum(A, A.T)

        A /= np.sum(A)  # Normalize

        b = np.random.randn(dim, 1)
        b = np.abs(b)  # Ensure positivity
        b /= np.sum(b)  # Normalize
        b = b.reshape(-1)

        return A, b

    def average_connectivity(self, A):
        num_nodes = A.shape[0]  # Number of nodes
        non_zero_counts = [np.count_nonzero(row)-1 for row in A]  # Count non-zero elements in each row
        avg_connectivity = sum(non_zero_counts) / num_nodes  # Calculate average connectivity
        return avg_connectivity

    def get_random_data(self, dim, sparcity_threshold=-0.25):
        """Return a sparse, symmetric, positive, normal distributed and normalized data matrix and observations vector"""
        A_rand = np.random.randn(dim, dim)
        # print(A_rand)
        A_sparce = A_rand * (A_rand < sparcity_threshold)
        # print(A_sparce)

        A = self.symmetric(np.triu(A_sparce)) + np.diag(np.diag(A_rand))
        A = np.abs(A)
        A = A / np.sum(A)

        b = np.abs(np.random.randn(dim, 1))
        b = b / np.sum(b)
        b = b.reshape(-1)

        return A, b

    def get_random_data(self, dim, sparcity_threshold=-0.25):
        """Return a sparse, symmetric, positive, normal distributed and normalized data matrix and observations vector"""
        A_rand = np.random.randn(dim, dim)
        print(A_rand)

        A_sparce = A_rand * (A_rand < sparcity_threshold)
        print(A_sparce)

        A = self.symmetric(np.triu(A_sparce)) + np.diag(np.diag(A_rand))
        A = np.abs(A)
        A = A / np.sum(A)

        b = np.abs(np.random.randn(dim, 1))
        b = b / np.sum(b)
        b = b.reshape(-1)

        return A, b
    
    def get_uniform_random_data(self, dim, target_k=3, target_percent=5, sparsity_threshold=1.0):
        """Return a sparse, symmetric, positive, uniformly distributed, and normalized data matrix and observations vector"""
        A_rand = np.random.random_sample((dim, dim))  # Generate matrix from uniform distribution [0, 1]
        A = self.symmetric(np.triu(A_rand)) + np.diag(np.diag(A_rand))  # Make symmetric and add diagonal
        A = A * (A_rand < sparsity_threshold)  # Apply sparsity threshold

        # Ensure at least one non-diagonal element
        while np.all(np.diag(A) == 1):
            r1, r2 = np.random.choice(range(dim), size=2, replace=False)
            A[r1, r2] = 0
            A[r2, r1] = 0

        # Ensure connectivity
        early_exit = False
        count = 0
        while not early_exit:
            r1, r2 = np.random.choice(range(dim), size=2, replace=False)
            if np.sum(A[r1]) > 0 and np.sum(A[:, r2]) > 0 and r1 != r2:
                A[r1, r2] = 0
                A[r2, r1] = 0
                if ((self.average_connectivity(A) - target_k) / target_k * 100 < target_percent):
                    early_exit = True
            else:
                count += 1
                if count > 10:
                    raise Exception("Early exit condition not met after 10 attempts")
        
        A = np.abs(A)
        A = A / np.sum(A)

        b = np.abs(np.random.randn(dim, 1))
        b = b / np.sum(b)
        b = b.reshape(-1)

        return A, b

    def symmetric(self, matrix):
        return matrix + matrix.T - np.diag(matrix.diagonal())

    def new_random_function(self, dim, density=0.25):
        A_rand = np.random.randn(dim, dim)
        num_to_replace = int(dim*density)

        for i in range(num_to_replace):
            indices_x = np.random.randint(0, A_rand.shape[0], num_to_replace)
            indices_y = np.random.randint(0, A_rand.shape[1], num_to_replace)
            A_rand[indices_x, indices_y] = 0

        A = np.abs(A_rand)
        A = A / np.sum(A)

        b = np.abs(np.random.randn(dim, 1))
        b = b / np.sum(b)
        b = b.reshape(-1)

        return A, b
    
# 1D Line Problem ---------------------------------
    def fetch_1D_line_padding(self, dim):
        if (dim == 1):
            return np.array([1])
        X = np.zeros((dim,dim))
        for i in range(dim):
            if (i == 0):
                X[i][i] = 1
                X[i+1][i] = 1
            elif (i == dim-1):
                X[i-1][i] = 1
                X[i][i] = 1
            else:
                X[i-1][i] = 1
                X[i][i] = 1
                X[i+1][i] = 1
        return X

    def get_1D_line_matrix(self, dim, normalized=True, scaling=False):
        A_rand = np.random.randn(dim, dim)
        A_sparce = A_rand * self.fetch_1D_line_padding(dim)

        A = self.symmetric(np.triu(A_sparce)) + np.diag(np.diag(A_rand))
        A = np.abs(A)
        
        if scaling:
            scaling_factors_A = np.random.randint(1, 1001, size=A.shape)  # Generate random scaling factors for each element in A
            A = A * scaling_factors_A
        
        b = np.abs(np.random.randn(dim, 1))
        
        if scaling:
            scaling_factors_b = np.random.randint(1, 1001, size=b.shape)  # Generate random scaling factors for each element in b
            b = b * scaling_factors_b
        
        b = b / np.sum(b) if normalized else b.reshape(-1)

        return A, b
# 2D Lattice Problem ---------------------------------
    def get_2D_lattice_padding(self, dim_x, dim_y):
        
        G=nx.grid_2d_graph(dim_x,dim_y)
        # nx.draw(G)
        # graph.draw_graph(title=f'Graph of 2D-Lattice {x, y}', color=colors[0])
        # plt.show()

        # A = nx.linalg.graphmatrix.adjacency_matrix(G, G.nodes())
        # # A = A.todense()
        # print(A)

        B=nx.adjacency_matrix(G,nodelist=sorted(G.nodes()))
        B = B.todense()
        # print(type(B))
        A = np.squeeze(np.asarray(B))
        # print(type(A))
        # print(A.shape)
        for i in range(A.shape[0]):
            A[i][i] = 1
        # print(A)
        return A
    
    def get_2D_lattice_matrix(self, dim_x, dim_y):
        number_nodes = dim_x * dim_y

        A_rand = np.random.randn(number_nodes, number_nodes)

        A_pad = self.get_2D_lattice_padding(dim_x,dim_y)

        A_sparce = A_rand * A_pad
        # print(A_sparce)

        A = self.symmetric(np.triu(A_sparce)) + np.diag(np.diag(A_rand))
        A = np.abs(A)

        b = np.abs(np.random.randn(number_nodes, 1))
        b = b / np.sum(b)
        b = b.reshape(-1)
        
        return A, b
# 2D Lattice Problem ---------------------------------

    def get_sparse_tree_matrix(self, dim, sparcity_threshold=-0.25):
        """Use minimum spanning tree to convert a sparse matrix for a sparse tree matrix"""
        A, b = self.get_uniform_random_data(dim, sparcity_threshold)
        A_diag = np.diag(np.diag(A))
        A_up_triangle = np.triu(A) - A_diag
        A_sparse = csr_matrix(A_up_triangle)
        A_min_span_tree = minimum_spanning_tree(A_sparse).toarray()
        A = A_min_span_tree + A_diag
        A = self.symmetric(A)
        return A, b

    @staticmethod
    def _add_loop_to_A(A, edge1, edge2):
        """Takes a tree graph and add edge between given nodes to get a loop"""
        edges = find_edges(A)
        random_edges = np.transpose([edges[edge1], edges[edge2]])

        for edge in random_edges:
            i, j = edge[0], edge[1]
            node_val = np.abs(np.random.randn(1)[0])
            A[i][j] = node_val
            A[j][i] = node_val
        return A

    def add_loops_to_A(self, A, max_loops):
        """create loops between already exist nodes (do nothing if already connected)"""
        num_nodes = A.shape[0]

        for i in range(max_loops):
            edge1 = np.random.randint(0, num_nodes - 1)
            edge2 = np.random.randint(0, num_nodes - 1)
            A = self._add_loop_to_A(A, edge1, edge2)

        return A

    @staticmethod
    def cut_random_edges(A, max_cut_edges):
        """cut random edges from a given graph matrix A"""
        num_nodes = A.shape[0]
        edges = find_edges(A)

        for _ in range(max_cut_edges):
            edge_idx = np.random.randint(0, num_nodes - 1)
            edge = edges[edge_idx]
            i, j = edge[0], edge[1]
            A[i][j] = 0
            A[j][i] = 0

        return A