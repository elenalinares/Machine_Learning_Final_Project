#This file is a Decision tree coded only with numpy, we're Nara Smith now

#We import the libraries first:
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
@dataclass


class _Node:
    is_leaf: bool
    prediction: float = None
    feature_idx: int = None
    threshold: float = None
    left:'_Node' = None
    right:'_Node' = None


class DecisionTreeRegressorScratch:
    """
    simple CART-style regression tree (MSE impurtity) implemented with NumPy only

    --------------------------

    We'll use these parameters:

    max_depth : int
        Maximum depth of our tree, >= 1
    
    min_samples_split : int
        Minimum sumber of samples required to split a node

    min_impurity_decrease : float
        Minimum required reduction in MSE to accept a split

    """
