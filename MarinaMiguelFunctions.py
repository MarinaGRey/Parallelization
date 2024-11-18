# Functions for First Practical Work (Massive Computing)
# Created by Marina Gómez Rey (100472836) and Miguel Fernández Lara (100473125). Group 96
# 20/10/2024



# Import relevant libraries
import numpy as np
from multiprocessing.sharedctypes import Value, Array, RawArray
from multiprocessing import Process, Lock
import ctypes



# Shared memory functions
def tonumpyarray(mp_arr): # Retrieve a multiprocess array with input type float 64. This should be enough to store large determinant values.
    return np.frombuffer(mp_arr.get_obj(),dtype=np.float64)

def init_sharedarray(shared_array,matrix_shape, mat): # Initialize the shared array with the global variables to be used throughout the project
    global shared_space 
    global shared_matrix
    global matrix
    matrix = mat
    shared_space = shared_array
    shared_matrix = tonumpyarray(shared_space).reshape(matrix_shape) # Reshape the shared memory array into matrix form for simplicity




# Calculating the minors using parallelization by rows to reduce overhead.
def calculate_minor_for_row(row_idx):
    
    frow = [0] * len(matrix) # Initialize a vector for the results of the determinants of the minors in each row.
    
    for j in range(len(matrix)): # Iterate through the matrix rows

        
        # Retrieve the minor matrix by removing the row and column of each specific cell. This is done using a vectorized operation that
        # has a low complexity, which means it runs in a extremely low amount of time
        minor_mat = [row[:j] + row[j+1:] for row in (matrix[:row_idx] + matrix[row_idx+1:])]

        # Now compute the determinant of the minor matrix. This is the part that takes up the most amount of time
        frow[j] = determinant(minor_mat)
        
    # We use locks in order to secure transactions and correct flow of data whithin the shared memory.
    with shared_space.get_lock(): # Initialize a lock per process when accessing the shared memory
        shared_matrix[row_idx, :] = frow  # Store the determinants vector inside the reshaped shared memory (no need to flatten it)


def determinant(mat): 
    
    # The determinant will be computed using the LU factorization. 
    # This function has a complexity of O(n^3). There's no other method with less complexitity to compute a determinant without numpy nor recursion.
    
    n = len(mat) # Get the length of the minor matrix. It will be (n-1)x(n-1), where n is the length of the original matrix.
    
    # Initialize L and U matrices
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]

    for i in range(n): # Iterate through the whole matrix
        
        # Upper triangular matrix U
        for k in range(i, n):
            
            # The goal is to make 0's below the main diagonal. This is done by finding the factor (which will be stored in the L matrix) that when 
            # multiplied with one of the rows below and substracted with the current row results in an exact 0.
            U[i][k] = mat[i][k] - sum(L[i][j] * U[j][k] for j in range(i)) 

        # Lower triangular matrix L, update it because it is needed for following iterations.
        for k in range(i, n):
            if i == k:
                L[i][i] = 1  # Diagonal of L is 1
            else:
                L[k][i] = (mat[k][i] - sum(L[k][j] * U[j][i] for j in range(i))) / U[i][i] # Update matrix with the factor that results in 0.
                
    det = 1 # Initialize the determinant
    
    for i in range(n):
        det *= U[i][i] # The determinant is the product of the diagonal elements of U

    return det




# The next functions are run after parallelizing.

def cofactor_matrix(matrix):
    
    # Just multiplying the matrix by 1 or -1
    
    n = len(matrix)
    cof = np.ones((n, n), dtype=int) # Build a matrix will all 1's
    
    for i in range(n):
        for j in range(n):
            if (i + j) % 2 != 0: # Every other position must be a -1. Just like a chess board
                cof[i, j] = -1
                
    return cof*matrix # Multiply element-wise the matrix of minors with the cofactor matrix


def transpose_matrix(matrix):
    
    transposed = [] # Initialize an empty list to store the transposed matrix
    
    for j in range(len(matrix[0])): # Iterate through the columns of the original matrix
        
        new_row = [] # Create a new row (list) for the transposed matrix
        
        for i in range(len(matrix)):
            new_row.append(matrix[i][j])  # Append the element at (i, j) to the new row
        transposed.append(new_row)  # Append the new row to the transposed matrix
    
    return transposed


def multiply_number_by_matrix(number, matrix): # This is done to multiply the inverse of the determinant by the resulting matrix
    
    result = [] # Initialize an empty list to store the result
    
    for row in matrix: # Iterate through each row of the matrix
        new_row = [] # Create a new row for the result
        for element in row: # Iterate through each element in the row 
            new_row.append(element * number) # Multiply the element by the number and append to the new row
            
        result.append(new_row) # Append the new row to the result matrix
    
    return result
