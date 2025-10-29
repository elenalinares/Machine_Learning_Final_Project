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

    def __init__(self, max_depth = 5, min_samples_split = 20, min_impurity_decrease = 1e-7, random_state =None):
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.min_impurity_decrease = float(min_impurity_decrease)
        if random_state is not None:
            np.random.seed(random_state)
        self.root_: _Node | None = None

# --- utilities -----------------------------------------------------------
    @staticmethod
    def _mse(y: np.ndarray) -> float:
        if y.size == 0:
            return 0.0
        
        #MSE of predictin the node mean

        mean = y.mean()
        return float(np.mean((y-mean)** 2))
    
    @staticmethod
    def _leaf(y: np.ndarray) -> _Node:
        #prediction is the mean target in the node
        return _Node(is_leaf = True, prediction= float(y.mean()) if y.size else 0.0)
    

# --- public API -----------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype = float)
        y = np.asarray(y, dtype = float).reshape (-1)
        assert X.shape[0] == y.shape[0], "X and y need to be the same lenght, same amount of rows"
        self.root_ = self._build_tree(X, y, depth = 0)
        return self