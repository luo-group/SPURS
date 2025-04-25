"""
This module implements functionality for identifying and analyzing functional sites in proteins
using SPURS model predictions and sigmoid-based regression analysis. It provides tools for
normalizing mutation effect scores, fitting sigmoid functions to the data, and visualizing
the results to identify functionally important positions in proteins.
"""

from spurs.inference import get_SPURS, parse_pdb
# ~ 10s
import torch
import pandas as pd
import numpy as np
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import seaborn as sns
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib as mpl

def normalize(data, lower=-4.36, upper=1.21, new_min=0, new_max=1):
    """
    Normalize mutation effect scores to a specified range.
    
    Args:
        data: Raw mutation effect scores
        lower: Lower bound for clipping (default: -4.36, 0.1 percentile of ddG values)
        upper: Upper bound for clipping (default: 1.21, 99.9 percentile of ddG values)
        new_min: Minimum value in normalized range (default: 0)
        new_max: Maximum value in normalized range (default: 1)
    
    Returns:
        Normalized and clipped data scaled to [new_min, new_max]
    """
    clipped_data = np.clip(data, lower, upper)
    norm_data = (clipped_data - lower) / (upper - lower)
    norm_data = norm_data * (new_max - new_min) + new_min
    return norm_data

def sigmoid_function(x, xmid, scal):
    """
    Compute sigmoid function values for given parameters.
    
    Args:
        x: Input values
        xmid: Midpoint parameter of the sigmoid
        scal: Scale parameter controlling steepness
    
    Returns:
        Sigmoid function values
    """
    return 1 / (1 + np.exp(-(x - xmid) / scal))

# Define a custom estimator for sigmoid fitting

class SigmoidRegressor(BaseEstimator, RegressorMixin):
    """
    Custom scikit-learn compatible estimator for fitting sigmoid functions to mutation effect data.
    Used to identify functional sites by finding positions where mutations have non-linear effects.
    
    The sigmoid function is fit using weighted least squares optimization with bounds
    derived from empirical protein stability data.
    """
    def __init__(self, xmid_initial=0.0, scal_initial=1.0):
        """
        Initialize sigmoid regressor with starting parameters.
        
        Args:
            xmid_initial: Initial guess for sigmoid midpoint
            scal_initial: Initial guess for sigmoid scale factor
        """
        self.xmid_initial = xmid_initial
        self.scal_initial = scal_initial

    def fit(self, X, y, sample_weight=None):
        """
        Fit sigmoid function to data using weighted optimization.
        
        Args:
            X: Input features
            y: Target values
            sample_weight: Sample weights for weighted least squares
        
        Returns:
            self: Fitted estimator
        """
        X, y = check_X_y(X, y, ensure_2d=False)
        
        # Initial guess for the parameters
        initial_guess = [self.xmid_initial, self.scal_initial]

        def weighted_loss(params, X, y, sample_weight):
            """Calculate weighted squared error loss"""
            xmid, scal = params
            y_pred = sigmoid_function(X, *params)
            
            residuals = y - y_pred

            return np.sum(sample_weight**2 * (residuals ** 2))

        # Optimize the parameters
        
        bounds = [(-20, -5), (None, np.e)]
# weight, bonds from https://github.com/lehner-lab/domainome/blob/main/09_esm1v_residuals.Rmd#L112
        # Optimize the parameters with bounds
        result = minimize(weighted_loss, initial_guess, args=(X, y, sample_weight), bounds=bounds)

        self.xmid_, self.scal_ = result.x
        
        return self

    def predict(self, X):
        """
        Predict using fitted sigmoid function.
        
        Args:
            X: Input features
        
        Returns:
            Predicted values
        """
        check_is_fitted(self, ['xmid_', 'scal_'])
        X = check_array(X, ensure_2d=False)
        return sigmoid_function(X, self.xmid_, self.scal_)
    
def get_sigmoid_results(mask_results,ddg):
    """
    Fit sigmoid function to mutation effect data for functional site identification.
    
    Args:
        mask_results: Mutation effect predictions from ESM
        ddg: Predicted stability measurements
    
    Returns:
        tuple: (normalized_ddg, mask_results, None, sigmoid_parameters)
    """
    X = mask_results.flatten()
    y = -normalize(ddg).flatten()
    assert len(X) == len(y)

    X_clean = X
    y_clean = y
    max_f_apca = np.nanmax(y)
    min_f_apca = np.nanmin(y)
    weights = max_f_apca - min_f_apca - (y_clean + 1)
    sigmoid_regressor = SigmoidRegressor(xmid_initial=np.nanmedian(X_clean), scal_initial=0.6)
    print(X_clean.shape,y_clean.shape,weights.shape)
    sigmoid_regressor.fit(X_clean.numpy(), y_clean.numpy(), sample_weight=weights.numpy())
    xmid = sigmoid_regressor.xmid_
    scal = sigmoid_regressor.scal_
    popt =  [xmid,scal]

    return y, X, None,popt

def inference_wt_seq(sequence: str, indices: list, batch_converter, model, device: torch.device, alphabet):
    """
    Perform inference on wild-type sequence using the ESM model.
    
    Args:
        sequence: Protein sequence to analyze
        indices: List of positions to analyze
        batch_converter: ESM batch converter
        model: ESM model
        device: Computation device
        alphabet: ESM alphabet for token mapping
    
    Returns:
        Logits for each position and possible amino acid substitution
    """
    data = [("sequence", sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
        logits = results["logits"]

    logits_original = logits[0, indices, :] 
    l = 'ACDEFGHIKLMNPQRSTVWY'
    logist = [logits_original[:,alphabet.tok_to_idx[i]] for i in l]
    return torch.stack(logist).T

def get_wt_aa_logit_differences(sequence: str, mut_indices: list, batch_converter, model, device: torch.device, alphabet, shift: int = 1):
    """
    Calculate differences between wild-type and mutant amino acid logits.
    
    This function helps identify positions where mutations would have the strongest effect
    by comparing model predictions for all possible amino acid substitutions against
    the wild-type amino acid at each position.
    
    Args:
        sequence: Protein sequence
        mut_indices: Positions to analyze
        batch_converter: ESM batch converter
        model: ESM model
        device: Computation device
        alphabet: ESM alphabet
        shift: Index adjustment (default: 1)
    
    Returns:
        Tensor of logit differences between wild-type and all possible mutations
    """
    mask_results = inference_wt_seq(sequence,[i for i in mut_indices],batch_converter,model,device,alphabet)
    
    l = 'ACDEFGHIKLMNPQRSTVWY'
    aa_indcies = [l.index(sequence[i-shift]) for i in mut_indices]
    # to one hot
    one_hot = torch.zeros(len(mut_indices),20).to(mask_results.device)
    one_hot[range(len(mut_indices)),aa_indcies] = 1
    wt_logits = (one_hot * mask_results).sum(-1).reshape(-1,1)
    # print(wt_logits.shape,mask_results.shape)
    mask_results = mask_results - wt_logits
    return mask_results

def plot_sigmoid_results(result,shift=1,vcenter = 0,highlight_positions =[]):
    """
    Visualize functional site prediction results using a scatter plot.
    
    Creates a plot showing normalized mutation effects across protein positions,
    with optional highlighting of specific positions of interest. The color scheme
    uses a diverging colormap centered at vcenter.
    
    Args:
        result: Tuple containing (normalized_ddg, mask_results, None, sigmoid_parameters)
        shift: Position numbering offset (default: 1)
        vcenter: Center value for color normalization (default: 0)
        highlight_positions: List of positions to highlight with markers
    
    Returns:
        normalized_data: Z-score normalized mutation effects
    """
    arr = (result[0].numpy()-sigmoid_function(result[1].numpy(), *result[-1])).reshape(-1,20).sum(-1)
    # normalize the data
    data = arr
    mean = np.nanmean(data)
    std_dev = np.nanstd(data)
    z_scores = (data - mean) / std_dev


    min_z = np.nanmin(z_scores)
    max_z = np.nanmax(z_scores)
    normalized_data = 2 * (z_scores - min_z) / (max_z - min_z) - 1


    x = np.arange(shift, shift + len(normalized_data))
    y = normalized_data

    color_strength = normalized_data
    vmin = min(color_strength)
    vmax = max(color_strength)
    

    norm = mcolors.TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=vcenter)
    cmap = plt.cm.get_cmap('RdBu_r')

    new_cmap = mpl.colors.LinearSegmentedColormap.from_list('truncated_cmap', cmap(np.linspace(0.1, 0.9, 256)))

    cmap = new_cmap
    plt.figure(figsize=(16, 4))
    scatter = plt.scatter(x, y, c=color_strength, cmap=cmap, norm=norm, s=100)

    cbar = plt.colorbar(scatter)


    color_strength = normalized_data

    
    for pos in highlight_positions:
        plt.scatter(pos, y[pos - shift], color='black', marker='x', s=120, linewidth=1.5)

    plt.ylim(-1.1, 1.1)

    plt.show()
    return normalized_data
