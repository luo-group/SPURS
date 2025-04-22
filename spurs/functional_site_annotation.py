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

def inference_org(sequence,index,batch_converter,model,device,alphabet):
    data = [("sequence", sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
        logits = results["logits"]

    logits_original = logits[0, index, :] 
    l = 'ACDEFGHIKLMNPQRSTVWY'
    logist = [logits_original[:,alphabet.tok_to_idx[i]] for i in l]
    return torch.stack(logist).T

def get_mask_results(sequence,mut_index,original_sequence,batch_converter,model,device,alphabet):
    mask_results = inference_org(original_sequence,[i for i in mut_index],batch_converter,model,device,alphabet)
    shift = 1
    l = 'ACDEFGHIKLMNPQRSTVWY'
    aa_indcies = [l.index(sequence[i-shift]) for i in mut_index]
    # to one hot
    one_hot = torch.zeros(len(mut_index),20).to(mask_results.device)
    one_hot[range(len(mut_index)),aa_indcies] = 1
    aa_dg = (one_hot * mask_results).sum(-1).reshape(-1,1)
    print(aa_dg.shape,mask_results.shape)
    mask_results = mask_results - aa_dg
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
