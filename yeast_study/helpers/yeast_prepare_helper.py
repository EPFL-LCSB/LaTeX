from pytfa.io.json import load_json_model

from skimpy.io.yaml import load_yaml_model
from skimpy.analysis.oracle.load_pytfa_solution import load_fluxes, \
    load_concentrations, load_equilibrium_constants
from skimpy.core.parameters import ParameterValuePopulation
from skimpy.sampling.simple_parameter_sampler import SimpleParameterSampler
from skimpy.sampling.ga_parameter_sampler import GaParameterSampler
from skimpy.utils.general import get_stoichiometry

from skimpy.analysis.mca.utils import get_dep_indep_vars_from_basis

from sympy import Matrix
from scipy.sparse import csc_matrix as sparse_matrix

import pandas as pd
import numpy as np
from sys import argv

def kinetic_model_prepare_ignore_int_issues(kmodel):
    """
    Only Zeus, the allmighty knows what this does. It's magic.
    """
    kmodel.prepare(mca=False)


    conservations = pd.read_csv(
        'latent_analysis/models/s_cerevisiae/steady_state_samples/magic_integer_stuff/ST10284_cons_relations_annotated.csv',
        index_col=0
    )


    # Transform the conservation relation to reflect dependent and independent weights
    L0, pivot = Matrix(conservations[kmodel.reactants].values).rref()
    kmodel.conservation_relation = sparse_matrix(L0, dtype=np.float)


    dep_ix, indep_ix = get_dep_indep_vars_from_basis(kmodel.conservation_relation)
    kmodel.independent_variables_ix = indep_ix
    kmodel.dependent_variables_ix = dep_ix
    kmodel.reduced_stoichiometry = get_stoichiometry(kmodel, kmodel.reactants)[indep_ix, :]

    return kmodel