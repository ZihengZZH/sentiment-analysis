import numpy as np
import math


def get_p_value(plus, minus, null):
    # para plus: #Plus
    # para minus: #Minus
    # para null: #Null
    N = 2*int(null/2)+plus+minus
    k = int(null/2) + min(plus, minus)
    q = .5
    p_val = .0
    for i in range(k):
        p_val += math.pow(q,N)*get_combination(N, i)
    print("p value is ", 2*p_val)
    return 2*p_val

def get_combination(upper, lower):
    return math.factorial(upper)/(math.factorial(lower)*math.factorial(upper-lower))
