#
# Utility functions for data processing for novel constrained sequential Latin Hypercube (with multidimensional uniformity) method (CASTRO)
# Jun 2024
# author: Christina Schenk

import numpy as np
from scipy.spatial.distance import cdist

def save_to_csv(filepath,
                samples):
    """
    writes samples to csv file

    Parameters
    ----------
    filepath: file path e.g. Path('Outputs/LHS_with_new_permutations_correct_suggestions_allselected_imp_ext_subprobs_improve.csv')
    samples: pandas dataframe with samples nsamp x ncomponents

    Returns
    -------
    csv file

    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    samples.to_csv(filepath)

    
#################################
###Synthesis constraints#########
#################################

def apply_single_synthesis_constraint(all_val_samples):
    """
    apply single synthesis constraints, sets maximum=1 and rest=0
    
    Parameters
    ----------
    all_val_samples: np array of samples of dimension number of points x number of components

    Returns
    -------
    0 1 samples

    """
    for i in range(len(all_val_samples)):
        j3 = np.where(all_val_samples[i,:]==all_val_samples[i,:].max())[0][0]
        all_val_samples[i,j3] = 1
        for k in range(np.shape(all_val_samples)[1]):
            if k!=j3:
                all_val_samples[i,k] = 0 
    return all_val_samples


def apply_mixed_synthesis_constraint(all_val_samples):
    """
    apply mixed synthesis constraints, components 0+2, 0+1, 1+2 in combination allowed, all as single componenent
    
    Parameters
    ----------
    all_val_samples: np array of samples of dimension number of points x number of components

    Returns
    -------
    samples fulfilling specific synthesis constraints

    """
    for i in range(len(all_val_samples)):
        # Set the element greater than 0.5 to 1 and others to 0
        if np.any(all_val_samples[i,:]>0.5):
            j3 = np.where(all_val_samples[i,:]>0.5)[0]
            all_val_samples[i,j3] = 1
            for k in range(np.shape(all_val_samples)[1]):
                if k!=j3:
                    all_val_samples[i,k] = 0 
        else:
            # Set the minimum value in the row to 0
            j3 = np.where(all_val_samples[i,:]==all_val_samples[i,:].min())[0][0]
            all_val_samples[i,j3] = 0
            # Normalize the remaining values to sum up to 1
            for k in range(np.shape(all_val_samples)[1]):
                if k!=j3:
                    all_val_samples[i,k] = all_val_samples[i,k]/np.sum(all_val_samples[i,:])#np.around(all_val_samples[i,k]/np.sum(all_val_samples[i,:]), decimals=ndec)
            # Find the minimum non-zero value's index (excluding index 3, here BN)
            j3 = np.where(all_val_samples[i,:]==all_val_samples[i,np.nonzero(all_val_samples[i,:])].min())[0][0]
            if j3!=3 and all_val_samples[i,j3]!=0:
                j3 = 3
            if j3==3 and all_val_samples[i,j3]==0:
                # Find the minimum non-zero value's index in the first two elements
                j3 = np.where(all_val_samples[i,0:2]==all_val_samples[i,np.nonzero(all_val_samples[i,0:2])].min())[0][0]
            # Set this index to 0
            all_val_samples[i,j3] = 0
            # Normalize the remaining values again to sum up to 1
            for k in range(np.shape(all_val_samples)[1]):
                if k!=j3:
                    all_val_samples[i,k] = all_val_samples[i,k]/np.sum(all_val_samples[i,:])#np.around(all_val_samples[i,k]/np.sum(all_val_samples[i,:]), decimals=ndec)        
    return all_val_samples   
    

def select_des_n_samp_random_pts(all_val_samples,
                                 des_n_samp=15):
    """
    choose des_n_samp random points from all samples
    
    Parameters
    ----------
    all_val_samples: np array of samples of dimension number of points x number of components
    des_n_samp: number of desired selected random points, default = 15

    Returns
    -------
    tol_samples: np array of reduced samples of length des_n_samp

    """
    number_of_rows = all_val_samples.shape[0] 
    random_indices = np.random.choice(number_of_rows,  
                                  size=des_n_samp)
    tol_samples = all_val_samples[random_indices]
    return tol_samples


def select_most_uniform_samples(samples, num_samples=90):
    """
    Select a subset of samples that maximizes uniformity using pairwise distance.

    Parameters:
    -----------
    samples : np.array
        Array of all samples (shape: n_samples x n_dimensions).
    num_samples : int
        Number of samples to select.

    Returns:
    --------
    uniform_samples : np.array
        Array of selected samples (shape: num_samples x n_dimensions).
    """
    # Initialize with a random sample
    np.random.seed(42)  # Set seed for reproducibility
    selected_indices = [np.random.choice(len(samples))]

    # Iteratively add points that maximize minimum pairwise distance
    while len(selected_indices) < num_samples:
        remaining_indices = [i for i in range(len(samples)) if i not in selected_indices]
        current_samples = samples[selected_indices]

        # Compute distances from remaining points to selected points
        distances = cdist(samples[remaining_indices], current_samples)

        # Find the point with the maximum minimum distance to selected points
        min_distances = np.min(distances, axis=1)
        best_index = remaining_indices[np.argmax(min_distances)]
        selected_indices.append(best_index)

    return samples[selected_indices]