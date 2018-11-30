import numpy as np
import math


# format () not C
def get_combination(upper, lower):
    return math.factorial(upper)/(math.factorial(lower)*math.factorial(upper-lower))


def get_p_value(plus, minus, null):
    # para plus: #Plus
    # para minus: #Minus
    # para null: #Null
    N = 2*math.ceil(null/2)+plus+minus
    k = math.ceil(null/2) + min(plus, minus)
    q = .5
    p_val = .0
    for i in range(k+1):
        p_val += get_combination(N, i)
    p_val *= math.pow(q,N)
    # two-sided value
    return p_val*2


def run_sign_test(result_A, result_B, feat_type):
    # para result_A: classification result of system A (without smoothing)
    # para result_B: classification result of system B (with smoothing)
    no_plus, no_minus, no_null = 0, 0, 0
    print("running sign testing ...")
    for i in range(len(result_A)):
        if result_A[i] > result_B[i]:
            no_plus += 1
        elif result_A[i] < result_B[i]:
            no_minus += 1
        else:
            no_null += 1
    p_value = get_p_value(no_plus, no_minus, no_null)

    f = open('./results_final.txt', 'a+', encoding='utf-8')
    f.write("\nSign test:\t Plus: %d, Minus: %d, Null: %d" % (no_plus, no_minus, no_null))
    f.write("\nthe p-value for this sign test between two classifier using %s features: %f" % (feat_type, p_value))
    f.close()
    print("\written to files ...")



'''
A LITTLE CONFUSED
The calculator in GraphPad used #'success' and #trails to calcualte p-value
It does not take the null/ties into account
However, disregarding ties will tend to affect a study's statistical power
Here we treat ties by adding 0.5 events to the positive and 0.5 events to the negative slide 
(and round up at the end)

'''