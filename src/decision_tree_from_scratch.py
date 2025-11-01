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
    simple CART (Classification And Regression Trees)-style regression tree (MSE impurtity) implemented with NumPy only --> at each split it picks the rule that reduces mse the most

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
    
    def predict(self, X:np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype = float)
        preds =np.array([self._predict_one(x, self.root_) for x in X], dtype =float)
        return preds
    
# --- core recursive builder ----------------------------------------------------
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth:int) -> _Node:
        #it stops when, reches end, too few samples, no useful split
        if (depth >= self.max_depth) or (X.shape[0] < self.min_samples_split):
            return self._leaf(y)
        
        best = self._best_split(X, y)
        if best is None or best["impurity_decrease"] < self.min_impurity_decrease:
            return self._leaf(y)
        
        f, thr = best["feature_idx"], best["threshold"]
        left_mask = X[:, f] < thr
        right_mask = ~ left_mask


        left = self._build_tree(X[left_mask], y[left_mask], depth +1)
        right = self._build_tree(X[right_mask], y[right_mask], depth +1)
        return _Node(is_leaf = False, feature_idx = f, threshold = thr, left= left, right = right)
            
    
    def _predict_one(self, x:np.ndarray, node: _Node) -> float:
        while not node.is_leaf:
            if x[node.feature_idx] < node.threshold:
                node =node.left

            else:
                node = node.right

        return node.prediction
    
# --- split search ----------------------------------------------------------------

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        """
        Try all features. For each feature, consider thresholds between sorted unique values.
        Return dict with best feature, threshold, and impurity decrease, or None if no split.
        """
        n_samples, n_features = X.shape
        parent_mse = self._mse(y)
        best = None

        for j in range(n_features):
            xj = X[:, j]
            order = np.argsort(xj, kind = "mergesort")
            x_sorted = xj[order]
            y_sorted = y[order]

            #possible thresholds are midpoints between distinct adjacent values
            distinct = np.where(np.diff(x_sorted) > 0)[0]
            if distinct.size == 0:
                continue

            #prefix sums to compute L/R means & MSE fast
            #cumulative sums and cumulative sums of squares
            csum = np.cumsum(y_sorted)
            csum2 = np.cumsum(y_sorted**2)

            for idx in distinct:
                nL = idx + 1
                nR = n_samples - nL

                #skip if one side would be too small
                if nL < self.min_samples_split or nR < self.min_samples_split:
                    continue


                sumL = csum[idx]
                sumR = csum[-1] - sumL
                sum2L = csum2[idx]
                sum2R = csum2[-1] - sum2L


                meanL = sumL/nL
                meanR = sumR/nR

                #MSE formula --> MSE = E[(y - mean)^2] = E[y^2] - mean^2
                mseL = (sum2L/nL) - (meanL** 2)
                mseR = (sum2R/nR) - (meanR **2)

                weighted = (nL/n_samples) * mseL + (nR / n_samples) * mseR
                impurity_decrease = parent_mse - weighted

                if (best is None) or (impurity_decrease > best["impurity_decrease"]):
                    thr = 0.5 * (x_sorted[idx] + x_sorted[idx + 1])
                    best = {
                        "feature_idx" : j,
                        "threshold": thr,
                        "impurity_decrease" : float(impurity_decrease),
                       
                    }

        return best
