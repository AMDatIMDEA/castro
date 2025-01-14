#
# Novel constrained sequential Latin Hypercube (with multidimensional uniformity) method
# Jan 2024
# author: Christina Schenk

#Python Packages:
import numpy as np
import pandas as pd
from scipy.stats import qmc
import lhsmdu
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sympy.utilities.iterables import multiset_permutations
from pathlib import Path  
#from itertools import permutations
import time
import random
import math
import itertools
from itertools import permutations
from scipy.spatial import minkowski_distance, distance_matrix
import csv
from sklearn.preprocessing import StandardScaler
import seaborn as sns

#---------------------------------------------------------------------------------------------------#
#### Latin Hypercube sampling on domain with variables summing up to 1: Sequentially and permutations such that condition holds (similar to Petelet et al (2010) but there for inequality constraints): 


def scale(sample,
          bounds):
    """ 
    scales the sample to be within the bounds
    
    Parameters
    ----------
    sample: vector of sample
    bounds: list of upper and lower bounds
    
    Returns
    -------
    scaled sample
    """
    return qmc.scale(sample, bounds[:][0], bounds[:][1])


def clip_and_scale(sample,
                   bounds):
    """
    clips and scales the sample to be within the bounds
    
    Parameters
    ----------
    sample: vector of sample
    bounds: list of upper and lower bounds
    
    Returns
    -------
    Clipped and scaled sample
    """
    return np.clip(sample, bounds[:][0], bounds[:][1])


def generate_lhs_sample(dimension,
                        bounds,
                        rs,
                        method="LHS",
                        n_samp=10):
    """
    generates LHS or LHSMDU sample
        
    Parameters
    ----------
    dimension: integer of dimension of input space
    bounds: list of upper and lower bounds
    rs: random seed, default 1234
    method: string of method: LHS or LHSMDU
    n_samp: number of samples to be generated, default is 10
    
    Returns
    -------
    scaled LHS or LHSMDU samples
    
    """
    if method == "LHS":
        sampler_new = qmc.LatinHypercube(d=dimension, seed=rs)
        sample = sampler_new.random(n=n_samp)
    elif method == "LHSMDU":
        sample = lhsmdu.sample(dimension, n_samp, randomSeed=rs).transpose()
    else:
        raise ValueError("Invalid method. Please specify 'LHS' or 'LHSMDU'.")

    return scale(sample, bounds)


def handle_dim_1(bounds,
                 method,
                 n_samp,
                 rs,
                 verbose=False):
    """
    handles sampling for dimension=1 
        
    Parameters
    ----------
    bounds: list of upper and lower bounds
    method: string of method: LHS or LHSMDU
    n_samp: number of samples to be generated
    rs: random seed, default 1234
    verbose: show prints and more info or not, default=False
    
    Returns
    -------
    feasible samples
    """
    x1_lb, x1_ub = bounds[0]
    sample1 = generate_lhs_sample(1, np.array([x1_lb, x1_ub]), rs,  method, n_samp)
    return sample1


def handle_dim_2(bounds,
                 method,
                 n_samp,
                 rs,
                 verbose=False):
    """
    handles sampling for dimension=2 
        
    Parameters
    ----------
    bounds: list of upper and lower bounds
    method: string of method: LHS or LHSMDU
    n_samp: number of samples to be generated
    rs: random seed, default 1234
    verbose: show prints and more info or not, default=False
    
    Returns
    -------
    feasible samples
    """
    x1_lb, x1_ub = bounds[0]
    x2_lb, x2_ub = bounds[1]

    sample1 = generate_lhs_sample(1, np.array([x1_lb, x1_ub]), rs, method, n_samp)
    sample2 = clip_and_scale(1.0 - sample1, np.array([x2_lb, x2_ub]))

    return sample1, sample2


def handle_dim_greater_than_2(bounds,
                              method,
                              n_samp,
                              max_iter,
                              max_iter_dim2,
                              max_iter_dim3,
                              max_rej,
                              rs,
                              verbose=False):
    """
    handles sampling for dimension>2 
        
    Parameters
    ----------
    bounds: list of upper and lower bounds
    method: string of method: LHS or LHSMDU
    n_samp: number of samples to be generated
    max_iter: maximum number of total iterations for dimension 1,2,3 to be feasible, if does not find minimum number of samples after this, breaks
    max_iter_dim2: maximum number of iterations for dimension 1,2 to be feasible
    max_iter_dim3: maximum number of iterations for dimension 1,2,3 to be feasible (for dimension=4)
    max_rej: integer number of maximum allowed samples to be rejected
    rs: random seed, default 1234
    verbose: show prints and more info or not, default=False
    
    Returns
    -------
    sample1, sample2, sample3: vectors
    """
    dim = len(bounds)
    start = time.time()
    infeasible = True
    l = 0
    l1 = 0
    x1_lb = bounds[0][0]
    x1_ub = bounds[0][1]
    x2_lb = bounds[1][0]
    x2_ub = bounds[1][1]

    while infeasible:
        if l > max_iter:
            print("No feasible sample found after " + str(l) + " iterations. Please increase max_rej, max_iter or check your bounds to find a feasible sample.")
        Sumsamp = np.zeros((n_samp,n_samp)) 
        C_1 = np.zeros((n_samp, n_samp))
        a_ind = []
        A = []
        B = []
        sample1 = generate_lhs_sample(1, np.array([x1_lb, x1_ub]), rs, method, n_samp)
        sample2 = generate_lhs_sample(1, np.array([x2_lb, x2_ub]), rs, method, n_samp)
        for i in range(len(sample1)):
            for k in range(len(sample2)):
                Sumsamp[i, k] = sum(sample1[i], sample2[k])
                if 0 < Sumsamp[i, k] <= 1.:
                    C_1[i, k] = 1
                    a_ind.append(tuple((i, k)))
                    if i not in A and k not in B:
                        A.append(i)
                        B.append(k)

        if len(A) == n_samp and len(B) == n_samp:
            sample1 = sample1[A]
            sample2 = sample2[B]
            sum_vec = np.add(sample1, sample2)
            infeasible = False
        elif len(A) != n_samp and len(B) != n_samp:
            if l1 > max_iter_dim2:
                if len(A) >= n_samp - max_rej and len(B) >= n_samp - max_rej:
                    sample1 = sample1[A]
                    sample2 = sample2[B]
                    sum_vec = np.add(sample1, sample2)
                    if verbose:
                        print(n_samp - len(A), "samples were rejected")
                    infeasible = False
                    n_samp = len(A)
            else:
                notA = []
                notB = []
                for ind in range(n_samp):
                    if ind not in A:
                        notA.append(ind)
                    if ind not in B:
                        notB.append(ind)
                count1=0
                counter_pair1=0
                for m in notA:
                    if counter_pair1>n_samp//4:
                        if verbose:
                            print("counter_pair exceeded")
                        break
                    if m not in list(zip(*a_ind))[0]:
                        count1+=1
                        #print("maybe need new sample or to reject")
                    if max_rej>0:
                        if count1>max_rej-1:
                            if verbose:
                                print("get new sample")
                            break

                    el0list = [ik for ik, (v,*_) in enumerate(a_ind) if v==m]
                    if len(el0list)>0:
                        #choose random tuple from that list:
                        chosentuple = np.random.choice(range(len(el0list)))

                        # if second index from random tuple not in D:
                        if a_ind[chosentuple][1] in B:
                            ind_rand2 = [i for i, x in enumerate(B) if x == a_ind[chosentuple][1]]
                            if len(ind_rand2)>0:
                                A.remove(A[ind_rand2[0]])
                                B.remove(a_ind[chosentuple][1])
                                notA.append(a_ind[chosentuple][0])
                                notB.append(a_ind[chosentuple][1])
                                A.append(m)
                                B.append(a_ind[chosentuple][1])
                                notA.remove(m)
                                notB.remove(a_ind[chosentuple][1])

                                counter_pair1+=1

                        else:
                            A.append(m)
                            B.append(a_ind[chosentuple][1])
                            notA.remove(m)
                            notB.remove(a_ind[chosentuple][1])

                        if len(A) == n_samp and len(B) == n_samp:
                            sample1 = sample1[A]
                            sample2 = sample2[B]
                            sum_vec = np.add(sample1, sample2)
                            infeasible = False
        
        l1 += 1

    del C_1
    del Sumsamp
    del a_ind
        
    if infeasible == False and dim==3:
        sample3 = np.zeros((len(sample1), 1))
        for i in range(len(sample1)):
            sample3[i] = 1.0-(sample1[i]+sample2[i])
            if np.any(sample3>bounds[2][1]) or np.any(sample3<bounds[2][0]):
                rej_list = [] 
                keep_list = []
                for i in range(len(sample3)):
                    if sample3[i]>bounds[2][1] or sample3[i]<bounds[2][0]:
                        if verbose:
                            print("reject because exceeding bounds")
                            print("index exceeding bounds", i)
                        rej_list.append(i)
                    else:  
                        keep_list.append(i)
                if len(rej_list)>n_samp-max_rej:
                    if verbose:
                        print("Warning:", len(rej_list), "samples were rejected because they were not within the bounds.") 
                sample1 = sample1[keep_list]
                sample2 = sample2[keep_list]
                sample3 = sample3[keep_list]
        del A
        del B
        if verbose:
            print("sample1", sample1)
            print("sample2", sample2)
            print("sample3", sample3)
            print("Sum all", sample1+sample2+sample3)

        end = time.time()
        if verbose:
            print("The conditioned " + str(method) + " algorithm took ", end-start, "CPUs")
        return sample1, sample2, sample3 
    
    if dim > 3:
        infeasible1, sample1, sample2, sample3 = handle_dim_greater_than_3(bounds, method, n_samp, max_iter, max_iter_dim3, max_rej, sum_vec, sample1, sample2, rs)
        l += 1
        if infeasible == False and infeasible1 == False:
            sample4 = np.zeros((len(sample1), 1))

            for i in range(len(sample4)):
                sample4[i] = 1.0-(sample1[i]+sample2[i]+sample3[i])

            if np.any(sample4>bounds[3][1]) or np.any(sample4<bounds[3][0]):
                rej_list = [] 
                keep_list = []
                for i in range(len(sample4)):
                    if sample4[i]>bounds[3][1] or sample4[i]<bounds[3][0]:
                        if verbose:
                            print("reject because exceeding bounds")
                            print("index exceeding bounds", i)
                        rej_list.append(i)
                    else:  
                        keep_list.append(i)
                if len(rej_list)>n_samp-max_rej:
                    if verbose:
                        print("Warning:", len(rej_list), "samples were rejected because they were not within the bounds.")
                sample1 = sample1[keep_list]
                sample2 = sample2[keep_list]
                sample3 = sample3[keep_list]
                sample4 = sample4[keep_list]
 
            if verbose:
                print("sample1", sample1)
                print("sample2", sample2)
                print("sample3", sample3)
                print("sample4", sample4)
                print("Sum all", sample1+sample2+sample3+sample4)

            end = time.time()
            if verbose:
                print("The conditioned " + str(method) + " algorithm took ", end-start, "CPUs")
            return sample1, sample2, sample3, sample4
        
        del C_2
        del Sumsamp2
        del A
        del B
        del C
        del D
        del c_ind


    
def handle_dim_greater_than_3(bounds,
                              method,
                              n_samp,
                              max_iter,
                              max_iter_dim3,
                              max_rej,
                              sum_vec,
                              sample1,
                              sample2,
                              rs,
                              verbose=False):
    """
    handles sampling for dimension>3
        
    Parameters
    ----------
    bounds: list of upper and lower bounds
    method: string of method: LHS or LHSMDU
    n_samp: number of samples to be generated
    max_iter: maximum number of total iterations for dimension 1,2,3 to be feasible, if does not find minimum number of samples after this, breaks
    max_iter_dim3: maximum number of iterations for dimension 1,2,3 to be feasible (for dimension=4)
    max_rej: integer number of maximum allowed samples to be rejected
    sum_vec: vector of sum of compatible sample1 and sample2
    sample1: vector of feasible sample1 
    sample2: vector of feasible sample1
    rs: random seed, default 1234
    verbose: show prints and more info or not, default=False
    
    Returns
    -------
    infeasible1: boolean flag
    sample1, sample2, sample3: vectors
    """
    
    x3_lb = bounds[2][0]
    x3_ub = bounds[2][1]
    infeasible1 = True
    l2 = 0
    
    while infeasible1:
        sample3 = generate_lhs_sample(1, np.array([x3_lb, x3_ub]), rs, method, n_samp)
        Sumsamp2 = np.zeros((n_samp,n_samp)) 
        C_2 = np.zeros((n_samp, n_samp))
        c_ind = []
        C=[]
        D=[]
        notC = []
        notD = []
        for i2 in range(len(sum_vec)):
            for k2 in range(len(sample3)):
                Sumsamp2[i2,k2] = sum_vec[i2]+sample3[k2]
                if Sumsamp2[i2,k2]<=1. and Sumsamp2[i2,k2]>0:
                    C_2[i2,k2] = 1
                    c_ind.append(tuple((i2, k2)))
                    if i2 not in C and k2 not in D:
                        C.append(i2)
                        D.append(k2)
        if len(C) == n_samp and len(D) == n_samp:
            sample1 = sample1[C]
            sample2 = sample2[C]
            sample3 = sample3[D]
            infeasible1 = False

        if len(C)!=n_samp and len(D)!=n_samp:
            if len(C)>=n_samp-max_rej and len(C)<n_samp:
                if verbose:
                    print("rejected", n_samp-len(C), " samples")
                sample1 = sample1[C]
                sample2 = sample2[C]
                sample3 = sample3[D]
                infeasible1 = False
                break
            if l2>max_iter_dim3:
                if len(C)>=n_samp-max_rej and len(D)>=n_samp-max_rej: #2*n_samp
                    sample1 = sample1[C]
                    sample2 = sample2[C]
                    sample3 = sample3[D]
                    infeasible1 = False
                else:
                    infeasible=True
                    print("No feasible sample found after " + str(l2) + " iteration. Please reinitiate sampling, increase max_rej, max_iter or check your bounds to find a feasible sample.")                    
                    break
            for k in range(n_samp):
                if k not in C:
                    notC.append(k)
                if k not in D:
                    notD.append(k)
            #check for feasible tuples
            count=0
            #check for missing C indices:
            counter_pair=0
            for m in notC:
                if counter_pair>n_samp:
                    if verbose:
                        print("counter_pair exceeded")
                    break
                if m not in list(zip(*c_ind))[0]:
                    count+=1
                        #print("maybe need new sample or to reject")
                if max_rej>0:
                    if count>max_rej-1:
                        if verbose:
                            print("get new sample")
                        break
                #list of possible tuples with current missing index m:
                el0list = [i for i, (v,*_) in enumerate(c_ind) if v==m]
                if len(el0list)>0:
                    #choose random tuple from that list:
                    chosentuple = np.random.choice(range(len(el0list)))

                    # if second index from random tuple not in D:
                    if c_ind[chosentuple][1] in D:
                        ind_rand2 = [i for i, x in enumerate(D) if x == c_ind[chosentuple][1]]
                        if len(ind_rand2)>0:
                            C.remove(C[ind_rand2[0]])
                            D.remove(c_ind[chosentuple][1])
                            notC.append(c_ind[chosentuple][0])
                            notD.append(c_ind[chosentuple][1])
                            C.append(m)
                            D.append(c_ind[chosentuple][1])
                            notC.remove(m)
                            notD.remove(c_ind[chosentuple][1])
                            counter_pair+=1
                    else:
                        C.append(m)
                        D.append(c_ind[chosentuple][1])
                        notC.remove(m)
                        notD.remove(c_ind[chosentuple][1])

                    if len(C) == n_samp and len(D) == n_samp:
                        sample1 = sample1[C]
                        sample2 = sample2[C]
                        sample3 = sample3[D]
                        infeasible1 = False
            
                    
        l2 += 1
        
    return infeasible1, sample1, sample2, sample3
                            
                        
def one_constrained_sampling(n_samp,
                             method="LHS",
                             bounds=None,
                             max_iter=60,
                             max_iter_dim2=60,
                             max_iter_dim3=60,
                             max_rej=None,
                             rs=None,
                             verbose=False):
    """
    one_constrained_sampling for up to 4 dimensions, where samples should add up to 1
        
    Parameters
    ----------
    n_samp: number of samples to be generated
    method: string of method: LHS or LHSMDU
    bounds: list of upper and lower bounds
    max_iter: maximum number of total iterations for dimension 1,2,3 to be feasible, if does not find minimum number of samples after this, breaks
    max_iter_dim2: maximum number of iterations for dimension 1,2 to be feasible
    max_iter_dim3: maximum number of iterations for dimension 1,2,3 to be feasible (for dimension=4)
    max_rej: integer number of maximum allowed samples to be rejected
    rs: random seed, default 1234
    verbose: show prints and more info or not, default=False
    
    Returns
    -------
    feasible samples from routines for appropriate number of dimensions
    """
    if max_rej is None:
        max_rej = n_samp//4
    if bounds is None or len(bounds) < 2:
        raise ValueError("Invalid bounds. Please provide valid bounds.")

    dim = len(bounds)

    if dim == 1:
        return handle_dim_1(bounds, method, n_samp, rs, verbose)

    elif dim == 2:
        return handle_dim_2(bounds, method, n_samp, rs, verbose)

    elif dim > 2:
        return handle_dim_greater_than_2(bounds, method, n_samp, max_iter, max_iter_dim2, max_iter_dim3, max_rej, rs, verbose)

    
#-------------------------------------------------------------------------------------------------#
### Collect samples for all permutations of bounds (all selected or subset of samples selected with checking for distance)

def get_bounds_for_dimension(combi,
                             prev_bounds):
    """
    extract bounds for specific dimension
        
    Parameters
    ----------
    combi: combination number of bounds, integer
    prev_bounds: list of upper and lower bounds
    
    Returns
    -------
    bounds for this combination number
    """
    return [prev_bounds[ind] for ind in combi]


# def stack_samples(samples,
#                   dim):
#     """
#     stacks samples
        
#     Parameters
#     ----------
#     samples: vectors of samples
#     dim: integer of dimension
    
#     Returns
#     -------
#     stacked samples
#     """
#     return np.column_stack(samples)[:dim]


def select_samples(all_samples,
                   val_samples_ord,
                   tol_norm,
                   num_select,
                   all_select):
    """
    select samples based on distance
        
    Parameters
    ----------
    all_samples: vectors of samples
    val_samples_ord: ordered samples in order of first bound permutation
    tol_norm: tolerance for minimum distance
    num_select: integer, number of samples to be selected if boolean=False
    all_select: boolean, if True all samples selected, if False num_select samples selected
    
    Returns
    -------
    selected samples
    """
    if all_select:
        return all_samples + val_samples_ord
    else:
        dist_matrix = distance_matrix(all_samples, val_samples_ord)
        selected_samples = []
        for j in range(len(val_samples_ord)):
            if np.all(dist_matrix[:, j]) > tol_norm:
                selected_samples.append(val_samples_ord[j])
                
        return all_samples + selected_samples[-num_select:]
    
    
def one_constrained_sampling_wrapper(methodname,
                                     dim,
                                     n_samp,
                                     bounds,
                                     max_rej,
                                     rs):
    """
    select samples based on distance
        
    Parameters
    ----------
    methodname: string of method, LHS or LHSMDU
    dim: dimension of input space
    n_samp: number of samples to be collected
    bounds: list of lower and upper bounds
    max_rej: maximum samples to be rejected
    rs: random seed, default 1234
    
    Returns
    -------
    one constrained feasible samples from routines for dim 2, 3, 4
    """
    if dim == 2:
        return one_constrained_sampling(n_samp, method=methodname, bounds=bounds, max_rej=max_rej, rs=rs)
    elif dim == 3:
        return one_constrained_sampling(n_samp, method=methodname, bounds=bounds, max_rej=max_rej, rs=rs)
    elif dim == 4:
        return one_constrained_sampling(n_samp, method=methodname, bounds=bounds, max_rej=max_rej, rs=rs)
    elif dim > 4:
        print("This algorithm works for maximum 4 dimensions.")
        return None

    
def sample_with_bound_permutations(
        prev_bounds, n_samp, tol_norm=1e-3, all_select=False, 
        num_select=4, max_rej=None, dim=None, rs=1234, verbose=False):
    """
    Computes samples with bound permutations.

    Parameters
    ----------
    prev_bounds : list
        List of lower and upper bounds for sampling.
    n_samp : int
        Number of samples to generate.
    tol_norm : float, optional
        Tolerance for minimum distance between samples.
    all_select : bool, optional
        Whether to select all samples or a fixed number.
    num_select : int, optional
        Number of samples to select if `all_select` is False.
    max_rej : int, optional
        Maximum number of rejections allowed during sampling.
    dim : int
        Dimensionality of the problem.
    rs : int, optional
        Random seed for reproducibility.
    verbose : bool, optional
        Whether to print verbose output.

    Returns
    -------
    tuple
        All feasible samples generated using two methods.
    """
    def get_permuted_bounds(combi, prev_bounds):
        return [prev_bounds[idx] for idx in combi]

    def stack_samples(samples, combi):
        # Reorder samples based on permutation combination
        stacked = np.zeros_like(samples)
        for num, ind in enumerate(combi):
            stacked[:, ind] = samples[:, num]
        return stacked

    def select_samps(existing_samples, new_samples, tol_norm, num_select):
        dist_matrix = distance_matrix(existing_samples, new_samples)
        selected_indices = [
            j for j in range(len(new_samples))
            if np.min(dist_matrix[:, j]) > tol_norm
        ]
        if not all_select:
            selected_indices = sorted(
                selected_indices, 
                key=lambda idx: np.min(dist_matrix[:, idx]), 
                reverse=True
            )[:num_select]
        return new_samples[selected_indices, :]

    # Initialize
    start = time.time()
    all_perms = list(permutations(range(dim)))
    if max_rej is None:
        max_rej = n_samp // 4
    all_val_samples, all_val_samples_mdu = None, None

    # Iterate over methods
    for num_meth in range(2):
        methodname = "LHS" if num_meth == 0 else "LHSMDU"
        for perm_ind, combi in enumerate(all_perms):
            bounds = get_permuted_bounds(combi, prev_bounds)
            if verbose:
                print(f"Permutation {perm_ind}: {combi}, Bounds: {bounds}")

            samples = one_constrained_sampling(
                method=methodname, n_samp=n_samp, bounds=bounds, max_rej=max_rej
            )

            # Convert samples to stacked format
            samples_stacked = np.column_stack(samples)
            if perm_ind == 0:
                if num_meth == 0:
                    all_val_samples = samples_stacked
                else:
                    all_val_samples_mdu = samples_stacked
            else:
                val_samples_ord = stack_samples(samples_stacked, combi)
                if num_meth == 0:
                    if all_select:
                        all_val_samples = np.vstack((all_val_samples, val_samples_ord))
                    else:
                        all_val_samples = np.vstack((
                            all_val_samples, 
                            select_samps(all_val_samples, val_samples_ord, tol_norm, num_select)
                        ))
                else:
                    if all_select:
                        all_val_samples_mdu = np.vstack((all_val_samples_mdu, val_samples_ord))
                    else:
                        all_val_samples_mdu = np.vstack((
                            all_val_samples_mdu, 
                            select_samps(all_val_samples_mdu, val_samples_ord, tol_norm, num_select)
                        ))

        if verbose:
            elapsed = time.time() - start
            print(f"{methodname} method completed in {elapsed:.2f} seconds.")

    return all_val_samples, all_val_samples_mdu
   
    
###Probabilistic version that runs sample_with_bound_permutations for multiple random seeds:
def prob_sample_with_bound_permutations(
    seeds=[42, 123, 7, 99, 56],
    prev_bounds=None, 
    n_samp=10, 
    tol_norm=1e-3, 
    all_select=False, 
    num_select=4, 
    max_rej=None, 
    dim=None, 
    verbose=False):
    """
    Perform probabilistic sampling with bound permutations for a list of seeds.
    
    Args:
        seeds (list): List of random seeds. by default list of 5 random seeds.
        prev_bounds (list): Previous bounds for sampling.
        n_samp (int): Number of samples per seed.
        tol_norm (float): Tolerance for the norm.
        all_select (bool): Whether to select all points.
        num_select (int): Number of selections to make.
        max_rej (int): Maximum rejections allowed.
        dim (int): Dimensionality of the sampling space.
        rs: random seed, default 1234
        verbose: show prints and more info or not, default=False
        
    Returns:
        tuple: Contains the selected LHS samples, selected MDU samples, and their means and standard deviations.
    """
    all_val_samples = []
    all_val_samples_mdu = []
    # Generate the samples for each seed
    for rs in seeds:
        all_val_samplesi, all_val_samples_mdui = sample_with_bound_permutations(
            prev_bounds=prev_bounds, n_samp=n_samp, tol_norm=tol_norm,
            all_select=all_select, num_select=num_select, max_rej=max_rej,
            dim=dim, rs=rs, verbose=False)
        all_val_samples.append(np.array(all_val_samplesi))  # Convert to NumPy array
        all_val_samples_mdu.append(np.array(all_val_samples_mdui))
    # Calculate the minimum length across all seeds
    min_len = min(len(samples) for samples in all_val_samples)
    min_len_mdu = min(len(samples) for samples in all_val_samples_mdu)

    # Initialize lists to store the sampled data
    all_val_samples_selected = []
    all_val_samples_mdu_selected = []

    # Randomly select `min_len` samples for each seed
    for k in range(len(seeds)):   
        # Randomly sample rows
        sampled_indices_lhs = np.random.choice(len(all_val_samples[k]), size=min_len, replace=False)
        sampled_indices_mdu = np.random.choice(len(all_val_samples_mdu[k]), size=min_len_mdu, replace=False)

        # Select the sampled rows
        sampled_lhs = all_val_samples[k][sampled_indices_lhs, :]
        sampled_mdu = all_val_samples_mdu[k][sampled_indices_mdu, :]

        # Append the sampled data to the lists
        all_val_samples_selected.append(sampled_lhs)
        all_val_samples_mdu_selected.append(sampled_mdu)

    # Stack the selected samples along a new axis (shape: num_seeds x min_len x n_dim)
    all_val_samples_selected = np.stack(all_val_samples_selected, axis=0)
    all_val_samples_mdu_selected = np.stack(all_val_samples_mdu_selected, axis=0)

    # Compute the mean and standard deviation across the seeds (axis=0)
    mean_samples = np.mean(all_val_samples_selected, axis=0)
    std_samples = np.std(all_val_samples_selected, axis=0)
    mean_samples_mdu = np.mean(all_val_samples_mdu_selected, axis=0)
    std_samples_mdu = np.std(all_val_samples_mdu_selected, axis=0)

    return (all_val_samples_selected, all_val_samples_mdu_selected, 
            mean_samples, std_samples, mean_samples_mdu, std_samples_mdu)


### Select subset of samples that varies the most in terms of distance from the already collected data
def scale_data(data,
               decimals=3):
    """
    scales data via standard scaling
        
    Parameters
    ----------
    data: numpy array or pandas dataframe of data
    decimals: decimals to be rounded to, integer
    
    Returns
    -------
    scaled data
    """
    scaler = StandardScaler().fit(data)
    return scaler.transform(np.around(data, decimals=decimals)), scaler


def select_samples(data_scaled,
                   samples,
                   samples_unscaled,
                   tol,
                   tol2,
                   decimals=3,
                   des_n_samp=None):
    """
    selects samples based on distance from data set
        
    Parameters
    ----------
    data_scaled: scaled data, numpy array
    samples: array of samples
    samples_unscaled: array of unscaled samples
    tol: tolerance for minimum distance to data, float
    tol2: tolerance for minimum distance to other already selected samples, float
    decimals: decimals to round to, integer
    des_n_samp: desired number of samples/experiments to be executed, default None
    
    Returns
    -------
    selected scaled rounded samples: array
    selected unscaled rounded samples: array
    selected_ind_list: list of selected indices
    """
    tollist = []
    norm_list = []

    for j in range(len(samples)):
        # Calculate the Euclidean distance between samples[j,:] and data_scaled
        distances = np.linalg.norm(data_scaled - samples[j, :], axis=1)
        if np.all(distances > tol):
            if len(tollist) == 0:
                tollist.append(j)
                norm_list.append(min([np.linalg.norm(data_scaled[i,:]-samples[j,:]) for i in range(len(data_scaled))]))
            else:
                # Calculate distances between samples[j,:] and samples in tollist
                distances_to_tollist = np.linalg.norm(samples[j, :] - samples[tollist, :], axis=1)
                if np.all(distances_to_tollist > tol2) and des_n_samp is not None and j not in tollist:
                    tollist.append(j)
                    norm_list.append(min([np.linalg.norm(data_scaled[i,:]-samples[j,:]) for i in range(len(data_scaled))]))

    sorted_norm_list = sorted(norm_list)
    selected_list = sorted_norm_list[-des_n_samp:]

    selected_norm_ind_list = [norm_list.index(i) for i in selected_list]
    selected_ind_list = [tollist[i] for i in selected_norm_ind_list]
    
    return np.around(samples[selected_ind_list, :],decimals=decimals), np.around(samples_unscaled[selected_ind_list,:], decimals=decimals), selected_ind_list


def select_samples_diff_from_data(exp_data,
                                  samples_LHS,
                                  samples_LHSMDU,
                                  des_n_samp=15,
                                  tol=5e-1,
                                  tol2=5e-1,
                                  decimals=3):
    """
    select samples based on distance from experimental data
        
    Parameters
    ----------
    exp_data: array of experimental data
    samples_LHS: LHS samples array
    samples_LHSMDU: LHSMDU samples array
    des_n_samp: desired number of samples/experiments to be executed
    tol: tolerance for minimum distance to data, float
    tol2: tolerance for minimum distance to other already selected samples, float
    decimals: decimals to round to, integer
    
    Returns
    -------
    tol_samples: selected samples with LHS
    tol_samples_LHSMDU: selected samples with LHSMDU
    tol_samples_unscaled: selected unscaled samples with LHS
    tol_samples_LHSMDU_unscaled: selected unscaled samples with LHSMDU
    
    """
    data_scaled, scaler = scale_data(exp_data)
    samples_LHS_scaled, scaler2 = scale_data(np.around(samples_LHS, decimals=decimals))
    samples_LHSMDU_scaled, scaler3 = scale_data(np.around(samples_LHSMDU, decimals=decimals))

    tol_samples, tol_samples_unscaled, selected_ind_list1 = select_samples(data_scaled, samples_LHS_scaled, samples_LHS, tol, tol2, decimals, des_n_samp)
    tol_samples_LHSMDU, tol_samples_LHSMDU_unscaled, selected_ind_list2 = select_samples(data_scaled, samples_LHSMDU_scaled, samples_LHSMDU, tol, tol2, decimals, des_n_samp)

    return tol_samples, tol_samples_LHSMDU, tol_samples_unscaled, tol_samples_LHSMDU_unscaled