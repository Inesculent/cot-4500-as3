import numpy as np


def function(t: float, w: float):
    return t - w**2


def eulers():
    
    original_w = 1
    start_of_t, end_of_t = (0, 2)
    num_of_iterations = 10

    # set up h
    h = (end_of_t - start_of_t) / num_of_iterations

    for cur_iteration in range(0, num_of_iterations):
        # do we have all values ready?
        t = start_of_t
        w = original_w
        h = h

        # create a function for the inner work

        # this gets the next approximation
        next_w = w + h * function(t,w)

        # we need to set the just solved "w" to be the original w
        # and not only that, we need to change t as well
        start_of_t = t + h
        original_w = next_w
        
    return original_w

def do_work(t, w, h):


    k1 = h * function(t,w)
    k2 = h * function(t+(h/2), w + (1/2) * k1)
    k3 = h * function(t + (h/2), w + (1/2) * k2)

    incremented_t = t + h

    k4 = h * function(incremented_t, w + k3)

    incremented_function_call = (1/6)*(k1 + 2*k2 + 2*k3 + k4)

    return incremented_function_call

def midpoint_method():
    original_w = 1
    start_of_t, end_of_t = (0, 2)
    num_of_iterations = 10

    next_w = 1

    # set up h
    h = (end_of_t - start_of_t) / num_of_iterations

    for cur_iteration in range(0, num_of_iterations):
        # do we have all values ready?
        t = start_of_t
        w = original_w
        h = h

        # # so now all values are ready, we do the method (THIS IS UGLY)
        # first_argument = t + (h / 2)
        # another_function_call = function(t, w)
        # second_argument = w + ( (h / 2) * another_function_call)
        # inner_function = function(first_argument, second_argument)
        # outer_function = h * (inner_function)

        # create a function for the inner work
        start_of_t = t + h
        inner_math = do_work(t, w, h)

        # this gets the next approximation
        next_w = w + inner_math


        # we need to set the just solved "w" to be the original w
        # and not only that, we need to change t as well
        
        original_w = next_w
        
    return next_w

def gauss_jordan(A, b):

    
    n = len(b)
    temp = np.empty([n])

    for i in range(n):
        temp[i] = A[0][i]
    
    # Combine A and b into augmented matrix
    Ab = np.concatenate((A, b.reshape(n,1)), axis=1)
    
    # Perform elimination
    for i in range(n):
        # Find pivot row
        max_row = i
        for j in range(i+1, n):
            if abs(Ab[j,i]) > abs(Ab[max_row,i]):
                max_row = j
        
        # Swap rows to bring pivot element to diagonal
        Ab[[i,max_row], :] = Ab[[max_row,i], :] # operation 1 of row operations
        
        # Divide pivot row by pivot element
        pivot = Ab[i,i]
        Ab[i,:] = Ab[i,:] / pivot
        
        # Eliminate entries below pivot
        for j in range(i+1, n):
            factor = Ab[j,i]
            Ab[j,:] -= factor * Ab[i,:] # operation 2 of row operations
    
    # Perform back-substitution
    for i in range(n-1, -1, -1):
        for j in range(i-1, -1, -1):
            factor = Ab[j,i]
            Ab[j,:] -= factor * Ab[i,:]
    
    # Extract solution vector x
    x = Ab[:,n]
    
    return temp




def UL(matrix):
    

    length = len(matrix)

    L = np.empty([length, length])

    for i in range (length):
        for j in range(length):

            if (i == j):
                L[i][j] = 1
            elif (j > i):
                L[i][j] = 0

    for i in range (length - 1):


        for j in range (length - 1):
            #If we encounter a j greater than i then we figure out a mult
            if (i < (j+1)):
                mult = matrix[j+1][i] / matrix[i][i]
                L[j+1][i] = mult
                #We then multiply said mult for all rows on i
                for k in range (length):
                    matrix[j+1][k] = matrix[j+1][k] - matrix[i][k]*mult
    

    determinant = 1

    for i in range (length):
        for j in range (length):
            if (i == j):
                determinant *= matrixU[i][j]

    print("\n%.5f" % determinant, "\n")
    
    print(L, "\n")
    return matrix

def U(matrix):
    

    length = len(matrix)


    for i in range (length - 1):


        for j in range (length - 1):
            #If we encounter a j greater than i then we figure out a mult
            if (i < (j+1)):
                mult = matrix[j+1][i] / matrix[i][i]
                #We then multiply said mult for all rows on i
                for k in range (length):
                    matrix[j+1][k] = matrix[j+1][k] - matrix[i][k]*mult
                    
    
    return matrix



def dom(matrix):

    length = len(matrix)

    arrayDiag = np.empty([length])
    arrayTotal = np.empty([length])

    for i in range (length):
        if (i == 0):
            total = matrix[i][1]
        else:
            total = matrix[i][0]
        
        for j in range (length):
            if (i == j):
                arrayDiag[i] = matrix[i][j]
                continue
            
            if (matrix[i][j] > 0):
                total += matrix[i][j]
            
        arrayTotal[i] = total
    


    flag = 1

    for i in range (length):
        if (arrayDiag[i] >= arrayTotal[i]):
            continue
        flag = 0
    
    if (flag == 0):
        print("False\n")
    else:
        print("True\n")


def posDefinite(matrix):

    flag = 1
    
    length = len(matrix)

    for z in range (length):

        innermatrix = np.empty([z,z])

        # fill matrix 

        for a in range (z):
            for b in range (z):
                innermatrix[a][b] = matrix[a][b]

        matrixU = U(innermatrix)

        determinant = 1

        for i in range (len(matrixU)):
            for j in range (len(matrixU)):
                if (i == j):
                    determinant *= matrixU[i][j]
        
        if (determinant < 0):
            flag = 0
            break
    
    if (flag == 1):
        print("True")
    else:
        print("False")






if __name__ == "__main__":
    np.set_printoptions(precision=7, suppress=True, linewidth=100)

    print("%.5f" % eulers(), "\n")

    print("%.5f" % midpoint_method(), "\n")

    A = np.array([[2,-1,1], [1,3,1],[-1,5,4]])
    b = np.array([6,0,3])

    print(gauss_jordan(A,b))


    matrixU = np.array([[1,1,0,3], 
            [2,1,-1,1], 
            [3,-1,-1,2], 
            [-1,2,3,-1]], dtype=np.double)
    

    matrixU = UL(matrixU)



    print(matrixU, "\n")

    matrix = np.array([[9,0,5,2,1], 
        [3,9,1,2,1], 
        [0,1,7,2,3], 
        [4,2,3,12,2],
        [3,2,4,0,8]], dtype=np.double)
    
    dom(matrix)
    
 
    matrix2 = np.array([[2,2,1], [2,3,0], [1, 0, 2]], dtype=np.double)

    posDefinite(matrix)



