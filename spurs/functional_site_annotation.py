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

    clipped_data = np.clip(data, lower, upper)

    norm_data = (clipped_data - lower) / (upper - lower)
    norm_data = norm_data * (new_max - new_min) + new_min
    return norm_data
def sigmoid_function(x, xmid, scal):
    return 1 / (1 + np.exp(-(x - xmid) / scal))
# Define a custom estimator for sigmoid fitting

class SigmoidRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, xmid_initial=0.0, scal_initial=1.0):
        self.xmid_initial = xmid_initial
        self.scal_initial = scal_initial

    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y, ensure_2d=False)
        
        # Initial guess for the parameters
        initial_guess = [self.xmid_initial, self.scal_initial]

        def weighted_loss(params, X, y, sample_weight):
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
        check_is_fitted(self, ['xmid_', 'scal_'])
        X = check_array(X, ensure_2d=False)
        return sigmoid_function(X, self.xmid_, self.scal_)
    
def get_sigmoid_results(mask_results,ddg):

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
    """Perform inference on a wild-type sequence using the ESM model.

    Args:
        sequence (str): The protein sequence to analyze.
        indices (list[int]): List of indices to analyze in the sequence.
        batch_converter: ESM batch converter for tokenizing sequences.
        batch_converter: ESM batch converter.
        model: ESM model.
        device: 'cpu' or 'cuda'

    Returns:
        torch.Tensor: Stacked logits for each position in the sequence, 
                     shape (len(indices), 20) where 20 represents the 20 standard amino acids.
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
    """Calculate differences between wild-type and all possible amino acid logits.

    For each position in the sequence, this function computes the difference between
    the wild-type amino acid logits and the logits for all 20 possible amino acids.
    This represents how much the model's predictions deviate from the wild-type at each position.

    Args:
        sequence (str): The protein sequence to analyze.
        mut_indices (list[int]): List of indices to analyze in the sequence.
        batch_converter: ESM batch converter.
        model: ESM model.
        device: 'cpu' or 'cuda'
        alphabet: ESM alphabet for token mapping.
        shift (int, optional): Shift value for index adjustment. Defaults to 1.

    Returns:
        torch.Tensor: Differences between wild-type and all possible amino acid logits,
                     shape (len(mut_indices), 20) where 20 represents the 20 standard amino acids.
                     Each value represents how much the model's prediction for that amino acid
                     differs from the wild-type at that position.
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
