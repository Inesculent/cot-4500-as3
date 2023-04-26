import numpy as np

def function(t: float, w: float):
    return t - w**2

def do_work(t, w, h):


    k1 = h * function(t,w)
    k2 = h * function(t+(h/2), w + (1/2) * k1)
    k3 = h * function(t + (h/2), w + (1/2) * k2)

    incremented_t = t + h

    k4 = h * function(incremented_t, w + k3)

    incremented_function_call = (1/6)*(k1 + 2*k2 + 2*k3 + k4)

    return incremented_function_call


def midpoint_method():
    original_w = 2.5
    start_of_t, end_of_t = (-1, 4)
    num_of_iterations = 30

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


if __name__ == "__main__":
    np.set_printoptions(precision=7, suppress=True, linewidth=100)

    print("%.5f" % midpoint_method(), "\n")
