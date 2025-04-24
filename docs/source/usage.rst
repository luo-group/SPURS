Usage Guide
===========

This guide will help you get started with using SPURS for protein stability prediction.

Basic Usage
----------
First, download the example PDB file from the SPURS repository:

.. code-block:: bash

   wget https://raw.githubusercontent.com/luo-group/SPURS/dev/data/inference_example/DOCK1_MOUSE.pdb

You can place this file in a ``data/inference_example/`` directory in your project.

Single Mutation Prediction
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from spurs.inference import get_SPURS, parse_pdb, get_SPURS_from_hub
   
   # Load the model
   model, cfg = get_SPURS_from_hub()
   
   # Prepare your protein data
   pdb_name = 'DOCK1_MOUSE'
   pdb_path = './data/inference_example/' + pdb_name + '.pdb'
   chain = 'A'
   pdb = parse_pdb(pdb_path, pdb_name, chain, cfg)
   
   # Make predictions
   ddg = model(pdb, return_logist=True)

The results are normalized, so the value in `ddg` for wild-type amino acids will be zero.

The model returns a tensor `ddg` containing stability predictions for all possible amino acid substitutions at each position. The values are normalized so that wild-type amino acids have a score of 0, while destabilizing mutations have positive scores and stabilizing mutations have negative scores.

To get the prediction for the wild-type amino acid at a specific position:

.. code-block:: python

   # wild-type amino acid at position 1
   wt_aa = pdb['seq'][0]
   ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
   ddg_wt = ddg[0,ALPHABET.index(wt_aa)]
   ddg_wt  # should be 0

For a specific mutation, like changing tryptophan at position 1 to alanine (W1A):

.. code-block:: python

   mt_aa = 'A'
   ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
   ddg_mt = ddg[0,ALPHABET.index(mt_aa)]
   ddg_mt  # ddg for W1A mutation

Multi-mutation Prediction
~~~~~~~~~~~~~~~~~~~~~~~

For predicting the effects of multiple mutations:

.. code-block:: python
   from spurs.inference import parse_pdb, get_SPURS_multi_from_hub, parse_pdb_for_mutation
   import torch

   # Define multiple mutations to analyze
   mut_info_list = [
       ['V2C','P3T'],  # First set of mutations
       ['W1A','V2Y'],  # Second set of mutations
   ]

   # Prepare protein data
   pdb_name = 'DOCK1_MOUSE'
   pdb_path = './data/inference_example/' + pdb_name + '.pdb'
   chain = 'A'
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

   # Load the multi-mutation model
   model, cfg = get_SPURS_multi_from_hub()

   # Parse PDB and prepare mutation data
   pdb = parse_pdb(pdb_path, pdb_name, chain, cfg)
   mut_ids, append_tensors = parse_pdb_for_mutation(mut_info_list)
   pdb['mut_ids'] = mut_ids
   pdb['append_tensors'] = append_tensors.to(device)

   # Make predictions
   ddg = model(pdb)
   # ddg[i] contains the prediction for mut_info_list[i]

The ``ddg`` tensor will contain stability predictions for each set of mutations in ``mut_info_list``. For example, ``ddg[0]`` corresponds to the combined effect of mutations V2C and P3T, while ``ddg[1]`` corresponds to W1A and V2Y mutations.



Functional Site Identification
----------------------------

SPURS can also be used to identify functional sites in proteins:

First, predict the stability of the mutations:
.. code-block:: python

   from spurs.inference import get_SPURS, parse_pdb, get_SPURS_from_hub
   # ~ 10s
   model, cfg = get_SPURS_from_hub()
   pdb_name = '1qlh'
   pdb_path = '../data/enzyme/1qlh.pdb'
   chain = 'A'
   pdb = parse_pdb(pdb_path, pdb_name, chain, cfg)
   # ~ 1s
   ddg = model(pdb,return_logist=True).cpu().detach()

Then, load esm and get the logit differences:

.. code-block:: python

   import esm
   import torch
   from spurs.functional_site_annotation import get_wt_aa_logit_differences

   ckpt = '../data/checkpoints/esm1v_t33_650M_UR90S_1/esm1v_t33_650M_UR90S_1.pt'
   model, alphabet = esm.pretrained.load_model_and_alphabet_local(ckpt)
   batch_converter = alphabet.get_batch_converter()
   model.eval()  
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   model = model.to(device)

   mut_index = list(range(2,376))
   '''
   mut_idx here is how the original_sequence aligned with the pdb['seq]
   for example, the original_sequence here is 'MSTAGKVIK...'
   and the pdb['seq'] is 'STAGKVIKCK...'
   so here original_sequence[2-1:376-1] shoudl align with pdb['seq']
   '''


   original_sequence =  'MSTAGKVIKCKAAVLWEEKKPFSIEEVEVAPPKAHEVRIKMVATGICRSDDHVVSGTLVTPLPVIAGHEAAGIVESIGEGVTTVRPGDKVIPLFTPQCGKCRVCKHPEGNFCLKNDLSMPRGTMQDGTSRFTCRGKPIHHFLGTSTFSQYTVVDEISVAKIDAASPLEKVCLIGCGFSTGYGSAVKVAKVTQGSTCAVFGLGGVGLSVIMGCKAAGAARIIGVDINKDKFAKAKEVGATECVNPQDYKKPIQEVLTEMSNGGVDFSFEVIGRLDTMVTALSCCQEAYGVSVIVGVPPDSQNLSMNPMLLLSGRTWKGAIFGGFKSKDSVPKLVADFMAKKFALDPLITHVLPFEKINEGFDLLRSGESIRTILTF'
   mask_results = get_wt_aa_logit_differences(original_sequence,mut_index,batch_converter,model,device,alphabet).cpu().detach()

Regression to Sigmoid and Plotting:

.. code-block:: python
   from spurs.functional_site_annotation import get_sigmoid_results

   result = get_sigmoid_results(mask_results,ddg)

   from spurs.functional_site_annotation import plot_sigmoid_results
   shift = 2
   vcenter = 0
   # ground truth label
   highlight_positions =[49] +[47,68,175]
   plot_sigmoid_results(result,shift,vcenter,highlight_positions)




Reproducing Results
-----------------

To reproduce the evaluation results from the paper:

.. code-block:: bash

   # For SPURS on Megascale and ten test sets
   python ./test.py experiment_path=data/checkpoints/spurs datamodule._target_=megascale data_split=test ckpt_path=best.ckpt mode=predict

   # For ThermoMPNN on Domainome
   python ./test.py experiment_path=data/checkpoints/ThermoMPNN datamodule._target_=domainome data_split=test ckpt_path=best.ckpt mode=predict

See Also
--------

- :doc:`api` for detailed API documentation
