import numpy as np


if __name__ == "__main__":
    a = np.array([[2,3,-1], [4,-2,1], [-2,1,2]])
    b = np.array([5,1,3])

    x = np.linalg.solve(a,b)

    print(x)


