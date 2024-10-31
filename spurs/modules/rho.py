import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from spurs import utils

from sklearn.metrics import roc_auc_score, average_precision_score
import torch
log = utils.get_logger(__name__)


def cal_roh(pred_scores,fermi_scores,pdb_chains,dataset_name):
    
    ret = {}
    pdb_chains_set = set(pdb_chains)
    pdb_chains = np.array(pdb_chains)
    
    dataset_name_set = set(dataset_name)
    dataset_name = np.array(dataset_name)
    
    pred_scores = pred_scores.cpu().numpy()
    fermi_scores = fermi_scores.cpu().numpy()
    for dataset in dataset_name_set:
        ret['avg_'+dataset] = {}
        mask = dataset_name == dataset
        srho = spearmanr(pred_scores[mask],fermi_scores[mask])[0]
        prho = pearsonr(pred_scores[mask],fermi_scores[mask])[0]
        r2 = r2_score(fermi_scores[mask],pred_scores[mask])
        mse = mean_squared_error(fermi_scores[mask],pred_scores[mask])
        
        pred_scores_bin = pred_scores[mask]>0
        fermi_scores_bin = fermi_scores[mask]>0
        auroc = roc_auc_score(fermi_scores_bin,pred_scores_bin)
        auprc = average_precision_score(fermi_scores_bin,pred_scores_bin)
        ret['avg_'+dataset] = {'spearman':srho,
                        'pearson':prho,
                        'r2':r2,
                        'mse':mse,
                        'rmse':np.sqrt(mse),
                        'auroc':auroc,
                        'auprc':auprc
                        }
    return ret

def cal_rho_by_chain(pred_scores,fermi_scores,pdb_chains,dataset_name,mut_ids,indices):
        ret = {}
        pdb_chains_set = set(pdb_chains)
        pdb_chains = np.array(pdb_chains)
        
        dataset_name_set = set(dataset_name)
        dataset_name = np.array(dataset_name)
        
        pred_scores = pred_scores.cpu().numpy()
        fermi_scores = fermi_scores.cpu().numpy()
        
        srho_list = []
        prho_list = []
        chain_list = []
        mse = mean_squared_error(fermi_scores,pred_scores)
        for chain in pdb_chains_set:
            ret[chain] = {}
            mask = pdb_chains == chain
            srho = spearmanr(pred_scores[mask],fermi_scores[mask])[0]
            prho = pearsonr(pred_scores[mask],fermi_scores[mask])[0]

            srho_list.append(srho)
            prho_list.append(prho)
            chain_list.append(chain)

        # breakpoint()
        # torch.save([pdb_chains,pred_scores,fermi_scores],'megascale_fig2e.pt')
        # torch.save([srho_list,chain_list,pred_scores,fermi_scores,pdb_chains,mut_ids,indices],'srho_list_chain_list_pdb_chains_pf_mut_id_indices.pt')
        ret = {'median_spearman':np.median(srho_list),
                        'median_pearson':np.median(prho_list),
                        # 'r2':r2,
                        'mse':mse,
                        # 'rmse':np.sqrt(mse),
                        # 'auroc':auroc,
                        # 'auprc':auprc
                        }
        return ret
