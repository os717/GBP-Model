import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import networkx as nx
import matplotlib as plt
import os

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

        A = A / np.sum(A)
        
        b = np.abs(np.random.randn(dim, 1))
        
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

        # b = b / np.sum(b)
        b = b.reshape(-1)
        
        return A, b
    
    def get_2D_lattice_matrix_PSD(self, dim_x, dim_y):
        number_nodes = dim_x * dim_y
        
        # Step 1: Generate a random matrix B
        B = np.random.randn(number_nodes, number_nodes)
        
        # Step 2: Construct A as B^T B to ensure A is positive semi-definite
        A = np.dot(B.T, B)
        
        # Step 3: Apply the lattice padding mask
        A_pad = self.get_2D_lattice_padding(dim_x, dim_y)
        A = A * A_pad
        
        # Step 4: Ensure symmetry and apply any necessary adjustments
        A = (A + A.T) / 2  # Symmetrize A
        
        b = np.abs(np.random.randn(number_nodes, 1))
        
        b = b.reshape(-1)
        
        return A, b
    
    def get_2D_lattice_matrix_PSD_TEST(self, dim_x, dim_y):
        number_nodes = dim_x * dim_y
        
        # Step 1: Generate a random matrix B
        B = np.random.randn(number_nodes, number_nodes)
        
        # Step 2: Construct A as B^T B to ensure A is positive semi-definite
        A = np.dot(B.T, B)
        
        # Step 3: Apply the lattice padding mask
        A_pad = self.get_2D_lattice_padding(dim_x, dim_y)
        A = A * A_pad
        
        # Step 4: Ensure symmetry and apply any necessary adjustments
        # A = (A + A.T) / 2  # Symmetrize A
        
        b = np.abs(np.random.randn(number_nodes, 1))
        
        b = b.reshape(-1)
        
        return A, b

    def get_2D_lattice_matrix_PSD_slow(self, dim_x, dim_y, perturbation_strength=0.01, rank=5, regularization_strength=1e-3):
        number_nodes = dim_x * dim_y
        
        # Step 1: Generate a random matrix B
        B = np.random.randn(number_nodes, number_nodes)
        
        # Step 2: Apply the lattice padding mask to B
        A_pad = self.get_2D_lattice_padding(dim_x, dim_y)
        B = B * A_pad  # Apply padding to B
        
        # Step 3: Construct A as B^T B to ensure A is positive semi-definite
        A = np.dot(B.T, B)
        
        # Step 4: Apply a low-rank perturbation to spread the eigenvalues
        C = np.random.randn(number_nodes, rank)  # Low-rank perturbation
        P = perturbation_strength * np.dot(C, C.T)
        
        A += P  # Add the low-rank perturbation to A
        
        # Step 5: Add a regularization term to maintain the condition number and prevent ill-conditioning
        A += regularization_strength * np.eye(number_nodes)
        
        # Step 6: Ensure symmetry (A should already be symmetric by construction, but for safety)
        A = (A + A.T) / 2  # Symmetrize A to handle any floating-point errors
        
        # Step 7: Create a random vector b
        b = np.abs(np.random.randn(number_nodes, 1))
        b = b.reshape(-1)
        
        return A, b
    
    def get_2D_lattice_matrix_PSD_shaped(self, dim_x, dim_y, eigenvalue_spread=10, regularization_strength=1e-4):
        number_nodes = dim_x * dim_y
        
        # Step 1: Generate a random matrix B
        B = np.random.randn(number_nodes, number_nodes)
        
        # Step 2: Apply the lattice padding mask to B
        A_pad = self.get_2D_lattice_padding(dim_x, dim_y)
        B = B * A_pad
        
        # Step 3: Construct A as B^T B to ensure A is positive semi-definite
        A = np.dot(B.T, B)
        
        # Apply lattice padding mask to keep the original zero entries
        A_pad = self.get_2D_lattice_padding(dim_x, dim_y)
        A = A * A_pad  # Ensure zero entries remain zero
        
        # Step 4: Perform eigenvalue decomposition
        eigvals, eigvecs = np.linalg.eigh(A)
        
        # Step 5: Modify the eigenvalue spectrum with a smaller spread
        eigvals = np.linspace(1, eigenvalue_spread, number_nodes)
        
        # Ensure no eigenvalues are too small
        eigvals = np.clip(eigvals, 1e-4, None)
        
        # Step 6: Reconstruct the matrix A with the modified eigenvalue spectrum
        A = eigvecs @ np.diag(eigvals) @ eigvecs.T
        
        # Step 7: Add minimal regularization to avoid ill-conditioning
        A += regularization_strength * np.eye(number_nodes)
        
        # Step 8: Apply lattice padding mask again to preserve zero entries
        A = A * A_pad  # Ensure zero entries remain zero
        
        # Symmetrize A to handle floating-point errors
        A = (A + A.T) / 2
        
        # Check condition number
        condition_number = np.linalg.cond(A)
        print(f"Condition number of A: {condition_number}")
        
        # Step 9: Create a random vector b
        b = np.abs(np.random.randn(number_nodes, 1))
        b = b.reshape(-1)
        
        return A, b
    
    def get_1D_line_matrix_PSD_difficult(self, dim, eigenvalue_spread=1e3, regularization_strength=1e-2, noise_strength=1e-2, show=False):
        number_nodes = dim
        
        # Step 1: Generate a random matrix B
        B = np.random.randn(number_nodes, number_nodes)
        
        # Step 2: Apply the lattice padding mask to B
        A_pad = self.fetch_1D_line_padding(number_nodes)
        B = B * A_pad
        
        # Step 3: Construct A as B^T B to ensure A is positive semi-definite
        A = np.dot(B.T, B)
        
        # Apply lattice padding mask to keep the original zero entries
        A = A * A_pad  # Ensure zero entries remain zero
        
        # Step 4: Perform eigenvalue decomposition
        eigvals, eigvecs = np.linalg.eigh(A)
        
        # Step 5: Modify the eigenvalue spectrum with a larger spread
        eigvals = np.linspace(1, eigenvalue_spread, number_nodes)
        
        # Ensure no eigenvalues are too small
        eigvals = np.clip(eigvals, 1e-2, None)
        
        # Step 6: Reconstruct the matrix A with the modified eigenvalue spectrum
        A = eigvecs @ np.diag(eigvals) @ eigvecs.T
        
        # Step 7: Add larger regularization to avoid ill-conditioning
        A += regularization_strength * np.eye(number_nodes)
        
        # Step 8: Apply lattice padding mask again to preserve zero entries
        A = A * A_pad  # Ensure zero entries remain zero
        
        # Step 9: Introduce random noise to the non-zero entries
        noise = noise_strength * np.random.randn(number_nodes, number_nodes)
        A += noise * A_pad  # Apply noise only where the mask is non-zero
        
        # Symmetrize A to handle floating-point errors
        A = (A + A.T) / 2
        
        # Check condition number
        condition_number = np.linalg.cond(A)
        if show:
            print(f"Condition number of A: {condition_number}")
        
        # Step 10: Create a random vector b
        b = np.abs(np.random.randn(number_nodes, 1))
        b = b.reshape(-1)
        
        return A, b

    
    def get_2D_lattice_matrix_PSD_difficult(self, dim_x, dim_y, eigenvalue_spread=1e3, regularization_strength=1e-2, noise_strength=1e-2, show=False):
        number_nodes = dim_x * dim_y
        
        # Step 1: Generate a random matrix B
        B = np.random.randn(number_nodes, number_nodes)
        
        # Step 2: Apply the lattice padding mask to B
        A_pad = self.get_2D_lattice_padding(dim_x, dim_y)
        B = B * A_pad
        
        # Step 3: Construct A as B^T B to ensure A is positive semi-definite
        A = np.dot(B.T, B)
        
        # Apply lattice padding mask to keep the original zero entries
        A = A * A_pad  # Ensure zero entries remain zero
        
        # Step 4: Perform eigenvalue decomposition
        eigvals, eigvecs = np.linalg.eigh(A)
        
        # Step 5: Modify the eigenvalue spectrum with a larger spread
        eigvals = np.linspace(1, eigenvalue_spread, number_nodes)
        
        # Ensure no eigenvalues are too small
        eigvals = np.clip(eigvals, 1e-2, None)
        
        # Step 6: Reconstruct the matrix A with the modified eigenvalue spectrum
        A = eigvecs @ np.diag(eigvals) @ eigvecs.T
        
        # Step 7: Add larger regularization to avoid ill-conditioning
        A += regularization_strength * np.eye(number_nodes)
        
        # Step 8: Apply lattice padding mask again to preserve zero entries
        A = A * A_pad  # Ensure zero entries remain zero
        
        # Step 9: Introduce random noise to the non-zero entries
        noise = noise_strength * np.random.randn(number_nodes, number_nodes)
        A += noise * A_pad  # Apply noise only where the mask is non-zero
        
        # Symmetrize A to handle floating-point errors
        A = (A + A.T) / 2
        
        # Check condition number
        condition_number = np.linalg.cond(A)
        if show:
            print(f"Condition number of A: {condition_number}")
        
        # Step 10: Create a random vector b
        b = np.abs(np.random.randn(number_nodes, 1))
        b = b.reshape(-1)
        
        return A, b

    def get_2D_lattice_matrix_PSD_difficult_target(self, dim_x, dim_y, target_condition_number=3000):
        number_nodes = dim_x * dim_y

        # Step 1: Generate a random matrix B
        B = np.random.randn(number_nodes, number_nodes)

        # Step 2: Apply the lattice padding mask to B
        A_pad = self.get_2D_lattice_padding(dim_x, dim_y)
        B = B * A_pad

        # Step 3: Construct A as B^T B to ensure A is positive semi-definite
        A = np.dot(B.T, B)

        # Apply lattice padding mask to keep the original zero entries
        A = A * A_pad  # Ensure zero entries remain zero

        # Step 4: Perform eigenvalue decomposition
        eigvals, eigvecs = np.linalg.eigh(A)

        # Step 5: Modify the eigenvalue spectrum to exactly achieve the target condition number
        max_eigenvalue = 1.0  # Set the largest eigenvalue to 1
        min_eigenvalue = max_eigenvalue / target_condition_number  # Set smallest eigenvalue to 1/3000

        # Manually set eigenvalues between min_eigenvalue and max_eigenvalue
        eigvals = np.linspace(min_eigenvalue, max_eigenvalue, number_nodes)

        # Step 6: Reconstruct the matrix A with the modified eigenvalue spectrum
        A = eigvecs @ np.diag(eigvals) @ eigvecs.T

        # Step 7: Apply lattice padding mask again to preserve zero entries
        A = A * A_pad  # Ensure zero entries remain zero

        # Step 8: Check the condition number
        condition_number = np.linalg.cond(A)
        print(f"Condition number of A: {condition_number}")

        # Step 9: Create a random vector b
        b = np.abs(np.random.randn(number_nodes, 1))
        b = b.reshape(-1)

        return A, b




    
    def get_2D_lattice_matrix_PSD_diagonal_dominant(self, dim_x, dim_y, dominance_factor=10):
        number_nodes = dim_x * dim_y
        
        # Step 1: Generate a random matrix B
        B = np.random.randn(number_nodes, number_nodes)
        
        # Step 2: Apply the lattice padding mask to B
        A_pad = self.get_2D_lattice_padding(dim_x, dim_y)
        B = B * A_pad  # Apply padding to B
        
        # Step 3: Construct A as B^T B to ensure A is positive semi-definite
        A = np.dot(B.T, B)
        
        # Step 4: Add diagonal dominance
        diag_values = np.abs(np.diag(A)) + dominance_factor
        np.fill_diagonal(A, diag_values)
        
        # Step 5: Ensure symmetry (A should already be symmetric by construction, but for safety)
        A = (A + A.T) / 2  # Symmetrize A to handle any floating-point errors
        
        # Step 6: Create a random vector b
        b = np.abs(np.random.randn(number_nodes, 1))
        b = b.reshape(-1)
        
        return A, b


    def get_1D_line_matrix_PSD(self, dim, scaling=False):
        number_nodes = dim * 1
        
        # Step 1: Generate a random matrix B
        B = np.random.randn(number_nodes, number_nodes)
        
        # Step 2: Construct A as B^T B to ensure A is positive semi-definite
        A = np.dot(B.T, B)
        
        # Step 3: Apply the lattice padding mask
        A_pad = self.fetch_1D_line_padding(dim)
        A = A * A_pad
        
        # Step 4: Ensure symmetry and apply any necessary adjustments
        A = (A + A.T) / 2  # Symmetrize A
        
        # Optional scaling
        if scaling:
            scaling_factors_A = np.random.randint(0, scaling, size=A.shape)
            A = A * scaling_factors_A

        if scaling:
            scaling_factors_b = np.random.randint(0, scaling, size=[A.shape[0], 1])
            b = scaling_factors_b
        else:
            b = np.abs(np.random.randn(number_nodes, 1))
        
        b = b.reshape(-1)
        
        return A, b
    
    def generate_SLAM_dataset_PSD(self, file_path):
        """Return a sparse, symmetric, positive, normal distributed and normalized data matrix and observations vector"""
        
        adjacency_matrix = self.get_SLAM_data_padding(file_path)
        dim = adjacency_matrix.shape[0]

        B = np.random.randn(dim, dim)

        # Step 2: Construct A as B^T B to ensure A is positive semi-definite
        A = np.dot(B.T, B)

        A_sparce = A * adjacency_matrix

        A = self.symmetric(np.triu(A_sparce)) + np.diag(np.diag(A))

        # Step 7: Generate observations vector b
        b = np.abs(np.random.randn(dim, 1))
        b = b.reshape(-1)

        # Step 8: Save files
        desired_part = file_path.split('/')[-1].rsplit('.', 1)[0]
        directory_path = os.path.join(os.path.dirname(os.path.dirname(file_path)), 'gbp_data')
        savefile_factor = os.path.join(directory_path, f'{desired_part}_factor_data.txt')
        savefile_marginal = os.path.join(directory_path, f'{desired_part}_marginal_data.txt')

        np.savetxt(savefile_factor, A)
        np.savetxt(savefile_marginal, b)
        
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
    

    def parse_g20_file(self, file_path):
        vertices = set()  # Use a set to ensure unique vertices
        edges = []

        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                parts = line.strip().split()
                if parts[0] == 'VERTEX_SE2':
                    vertex_id = int(parts[1])
                    vertices.add(vertex_id)
                elif parts[0] == 'EDGE_SE2':
                    vertex1 = int(parts[1])
                    vertex2 = int(parts[2])
                    edges.append((vertex1, vertex2))

        return sorted(vertices), edges  # Sorting to maintain order

    def create_adjacency_matrix(self, vertices, edges):
        n = len(vertices)
        adjacency_matrix = np.zeros((n, n), dtype=int)

        vertex_index = {vertex: idx for idx, vertex in enumerate(vertices)}

        for edge in edges:
            vertex1, vertex2 = edge
            idx1 = vertex_index[vertex1]
            idx2 = vertex_index[vertex2]
            adjacency_matrix[idx1][idx2] = 1
            adjacency_matrix[idx2][idx1] = 1  # Assuming the graph is undirected

        return adjacency_matrix

    def get_SLAM_data_padding(self, filename):
        vertices, edges = self.parse_g20_file(filename)
        adjacency_matrix = self.create_adjacency_matrix(vertices, edges)
        return adjacency_matrix

    def generate_SLAM_dataset(self, file_path):
        """Return a sparse, symmetric, positive, normal distributed and normalized data matrix and observations vector"""
        
        adjacency_matrix = self.get_SLAM_data_padding(file_path)
        
        dim = adjacency_matrix.shape[0]

        A_rand = np.random.randn(dim, dim)

        A_sparce = A_rand * adjacency_matrix

        A = self.symmetric(np.triu(A_sparce)) + np.diag(np.diag(A_rand))
        A = np.abs(A)

        b = np.abs(np.random.randn(dim, 1))
        b = b / np.sum(b)
        b = b.reshape(-1)

        # extract file
        desired_part = file_path.split('/')[-1].rsplit('.', 1)[0]
        directory_path = os.path.join(os.path.dirname(os.path.dirname(file_path)), 'gbp_data')
        savefile = os.path.join(directory_path, desired_part)

        np.savetxt(f'{savefile}_factor_data.txt', A)
        np.savetxt(f'{savefile}_marginal_data.txt', b)
        
        return
    
    def fetch_SLAM_dataset(self, file_path_factor, file_path_marginal):

        A = np.loadtxt(file_path_factor)
        b = np.loadtxt(file_path_marginal)

        return A, b