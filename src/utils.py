#
# Utility functions for data processing for novel constrained sequential Latin Hypercube (with multidimensional uniformity) method
# Jun 2024
# author: Christina Schenk

def save_to_csv(filepath, samples):
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
