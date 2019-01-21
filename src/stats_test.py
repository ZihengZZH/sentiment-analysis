import numpy as np
import math
from itertools import permutations


# return the combination value given N, i
def get_combination(upper, lower):
    return math.factorial(upper)/(math.factorial(lower)*math.factorial(upper-lower))


# calculate p-value for one specific pair of classifiers
def get_p_value(plus, minus, null, ignore_ties=False):
    # para plus: #Plus
    # para minus: #Minus
    # para null: #Null
    N, k = .0, .0
    q = .5
    p_val = .0

    if ignore_ties:
        N = plus + minus
        k = min(plus, minus)
    else:
        N = 2*math.ceil(null/2) + plus + minus
        k = math.ceil(null/2) + min(plus, minus)
    
    for i in range(k+1):
        p_val += get_combination(N, i)
    p_val *= math.pow(q,N)
    # two-sided value
    return p_val*2


# run the sign test given two lists of results 
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

    f = open('./results/results_final.txt', 'a+', encoding='utf-8')
    f.write("\nSign test:\t Plus: %d, Minus: %d, Null: %d" % (no_plus, no_minus, no_null))
    f.write("\nthe p-value for this sign test between two classifier using %s features: %f" % (feat_type, p_value))
    f.close()
    print("\written to files ...")


# return the permutation list (0/1 at each position)
def get_permutation_list(N):
    # para N: number of pairs of samples
    permu_list = set()
    for i in range(N+1):
        list_2_permu = [0]*i + [1]*(N - i)
        for temp_list in permutations(list_2_permu):
            permu_list.add(temp_list)
    return list(permu_list)


# calculate the mean difference between two given list and a swap list
def calc_mean_difference(list_A, list_B, swap_list):
    # para list_A:
    # para list_B:
    # para swap_list: 
    mean_diff_A, mean_diff_B = .0, .0
    length_n = len(list_A)
    assert len(list_A) == len(list_B), "ERROR! LENGTHS MISMATCH"
    assert len(list_B) == len(swap_list), "ERROR! LENGTHS MISMATCH"

    for i in range(len(swap_list)):
        if swap_list[i] == 0:
            mean_diff_A += list_A[i] / length_n
            mean_diff_B += list_B[i] / length_n
        else:
            mean_diff_A += list_B[i] / length_n
            mean_diff_B += list_A[i] / length_n
    return abs(round((mean_diff_B - mean_diff_A), 4))


# run Monte Carlo Permutation test
def run_permutation_test(result_A, result_B, R=0):
    # para result_A: np.array
    # para result_B: np.array
    # para R: preset number of permuted samples
    # para R: if R != 0, called Monte Carlo Permutation test
    p_value, no_larger = .0, 0
    assert len(result_A) == len(result_B), "ERROR! LENGTHS MISMATCH"

    permutation_list = get_permutation_list(len(result_A))
    original_result = calc_mean_difference(result_A, result_B, [0]*len(result_A))

    if R == 0:
        for i in range(len(permutation_list)):
            no_larger += calc_mean_difference(result_A, result_B, permutation_list[i]) >= original_result
        p_value = (no_larger + 1) / (len(permutation_list) + 1)
    else:
        for j in range(R):
            np.random.shuffle(permutation_list)
            no_larger += calc_mean_difference(result_A, result_B, permutation_list[j]) >= original_result
        p_value = (no_larger + 1) / (R + 1)

    return p_value, no_larger


'''

The calculator in GraphPad used #'success' and #trails to calcualte p-value
It does not take the null/ties into account (JUST IGNORE THE TIES)

However, disregarding ties will tend to affect a study's statistical power
Here we treat ties by adding 0.5 events to the positive and 0.5 events to the negative slide (and round up at the end). Therefore, the p-value could be a little different that calculated from the QuickCalcs or read from the Table D.

The sign test is quite straightforward:
    pre-select significance level alpha = .05 (5%) or .01 (1%)
    calculate p-value ---> if p-value <= 0.05 (> 0.05)
                      ---> reject null hypothesis 
                      ---> significant difference found 

NOTE that the p-value calculation in permutation test is as follows:
    p = (s+1) / (R+1) 
where s is the number of permuted samples with difference in M higher than one observed in the original runs, and R is the number of permutations. 

Furthermore, if R < 2^n because of size, we call this Monte Carlo Permutation test.

NOTE that given the monte-carlo nature of the algorithm you will not get exact same number on each run:

'''