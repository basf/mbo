"""Purpose: store pre-processing utilities. At the moment, these are functionality for (fuzzy) clustering of output columns by shared nans.
Ideally, would also have functionality for threshhold hypertuning."""
#################################
from copy import deepcopy
from functools import reduce
from typing import List

import numpy as np
import opti
import pandas as pd
from opti import Objectives, Parameters


def jaccard_sim(compare: list):
    """
    Jaccard similarity is a metric for measuring set similarity. Size of intersection over size of union.
    Returns a value between 0 and 1. 1 is identical. 0 is no overlap.
    https://effectivesoftwaredesign.com/2019/02/27/data-science-set-similarity-metrics/
    Args:
        compare: list of sets to calculate the jaccard similarity of. Can take arbitrary length.

    Returns: decimal between 0 and 1, jaccard similarity measure.

    """
    if len(compare) <= 1:
        return 1
    else:
        intersected = reduce(set.intersection, compare)
        unioned = reduce(set.union, compare)
        return len(intersected) / len(unioned)


def fuzzy_group_by_nans(df: pd.DataFrame, th=0.7):
    """Inputs dataframe. Clusters columns by shared non-nan indices.
    possible desired future functionality: fuzzy cluster, where matching percentage to be specified.

    Args:
        df (pd.DataFrame): dataframe to consider
        th (float): threshhold for jaccard similarity. Default 0.7

    Returns:
        [groups, grouped_cols]: groups is a list of unique sets of non-nan indices,
        grouped_cols is a list of lists. Each list are column names with similar non-nan indices, measured with jaccard_sim and threshold th.
        The relationship is such that groups[i] correspond to column names listed in grouped_cols[i]
    """
    groups = []
    grouped_cols = []

    for c in df.columns:
        add_new_group = True
        c_nnans = set(np.where(df[c].notnull())[0])
        for idx, g in enumerate(groups):
            # use jaccard_sim to check if other columns similar enough
            if jaccard_sim([g, c_nnans]) >= th:
                # if already present, no need to add to list
                add_new_group = False
                # add name of column to others with this non-nan set
                grouped_cols[idx].append(c)
                # replace nan set with intersection
                groups[idx] = g.intersection(c_nnans)
                break
        if add_new_group:
            # add set of not nans, add col name with next index number
            groups.append(c_nnans)
            grouped_cols.append([c])

    return groups, grouped_cols


def strict_group_by_nans(df: pd.DataFrame):
    """Inputs dataframe. Clusters columns by shared non-nan indices.
    Args:
        df (pd.DataFrame): dataframe to consider

    Returns:
        [groups, grouped_cols]: groups is a list of unique sets of non-nan indices,
        grouped_cols is a list of lists. Each list are column names with the same non-nan indices.
        The relationship is such that groups[i] correspond to column names listed in grouped_cols[i]
    """
    return fuzzy_group_by_nans(df=df, th=1)


def basic_subproblem(
    problem: opti.Problem,
    objective_names: List[str] = [],
    output_names: List[str] = [],
    name=None,
):
    """Draft of functionality that should move to mopti. I am probably forgetting something, but this "basic" subsetting is enough for what I need.

    Args:
        problem (opti.Problem): problem to subset
        output_names (List[str]): list of output names
    Returns:
        opti.Problem: subproblem. Might be wonky.
    """
    # check if *_names is a subset of problem.*.names:
    if set(problem.outputs.names).intersection(set(output_names)) != set(output_names):
        print(
            "output_names not a subset of the outputs of the problem. Returning problem"
        )
        return problem
    if set(problem.objectives.names).intersection(set(objective_names)) != set(
        objective_names
    ):
        print(
            "objective_names not a subset of the objectives of the problem. Returning problem"
        )
        return problem

    # deepcopy to modify
    sub_problem = deepcopy(problem)
    # "intersect". Made harder by all the classes and typing.

    if len(output_names) > 0:
        sub_problem.outputs = Parameters(
            [o for o in problem.outputs if o.name in output_names]
        )
        outs = output_names
    else:
        # sub_problem has same outputs, and
        outs = problem.outputs.names
    if len(objective_names) > 0:
        sub_problem.objectives = Objectives(
            [o for o in problem.objectives if o.name in objective_names]
        )
    else:
        # need to make sure it is at worst the outputs. Can have outputs but no objectives specified
        sub_problem.objectives = Objectives(
            [
                o
                for o in problem.objectives
                if o.name in set(problem.outputs.names).intersection(outs)
            ]
        )

    # option to change name
    if name is not None:
        sub_problem.name = name
    return sub_problem
