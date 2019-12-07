#!/usr/bin/env python

# ============================================================================
# Created by Kirstie Whitaker
# at Hot Numbers coffee shop on Trumpington Road in Cambridge, September 2016
# Contact: kw401@cam.ac.uk
# ============================================================================

# ============================================================================
# IMPORTS
# ============================================================================

import scona.make_corr_matrices as mcm
from scona.scripts.useful_functions import read_in_data
from scona.wrappers.parsers import corrmat_from_regionalmeasures_parser

def corrmat_from_regionalmeasures(regional_measures_file,
                                  names_file,
                                  output_name=None,
                                  covars_file=None,
                                  method='pearson'):
    '''
    Read in regional measures, names and covariates files to compute
    correlation matrix and write it to output_name.

    Parameters:
        * regional_measures_file : a csv containing data for some regional
            measures with brain regions and covariates as columns and subjects
            as rows. The first row of regional_measures should be column
            headings.
        * names_file : a text file containing names of brain regions. One name
            per line of text. These names key columns in df to correlate over.
        * covars_file : a text file containing a list of covariates to account
            for. One covariate per line of text. These names key columns in df.
        * output_name : file name to save output matrix to.
    '''
    # Read in the data
    df, names, covars_list, *a = read_in_data(
        regional_measures_file,
        names_file,
        covars_file=covars_file)

    M = mcm.corrmat_from_regionalmeasures(
        df, names, covars=covars_list, method=method)

    # Save the matrix
    if output_name is not None:
        mcm.save_mat(M, output_name)

    return M
    

def main():
    # Read in the command line arguments
    arg = corrmat_from_regionalmeasures_parser.parse_args()

    # Now run the main function :)
    corrmat_from_regionalmeasures(
        arg.regional_measures_file,
        arg.names_file,
        output_name=arg.output_name,
        covars_file=arg.covars_file,
        method=arg.method)

    
if __name__ == "__main__":
    main()
