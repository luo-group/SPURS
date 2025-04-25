API Reference
================

This page provides detailed documentation for the SPURS package. SPURS offers APIs for protein stability prediction and functional site identification.

Getting Started
-------------------------------------------------------------------------------------------

Model Loading and Inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: spurs.inference
   :members: get_SPURS, get_SPURS_from_hub, parse_pdb
   :undoc-members:
   :show-inheritance:

Basic usage:

.. code-block:: python

    from spurs.inference import get_SPURS_from_hub, parse_pdb

    # Load pre-trained model
    model, cfg = get_SPURS_from_hub()

    # Parse PDB file
    pdb = parse_pdb(
        pdb_path='path/to/protein.pdb',
        pdb_name='PROTEIN_NAME',
        chain='A',
        cfg=cfg
    )

    # Make predictions
    ddg = model(pdb, return_logist=True)

Functional Site Identification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: spurs.functional_site_annotation
   :members: get_wt_aa_logit_differences, inference_wt_seq
   :undoc-members:
   :show-inheritance:

Basic usage:

.. code-block:: python

    from spurs.functional_site_annotation import get_wt_aa_logit_differences
    import torch

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sequence = "YOUR_PROTEIN_SEQUENCE"
    mut_indices = list(range(1, len(sequence) + 1))

    # Get predictions
    results = get_wt_aa_logit_differences(
        sequence,
        mut_indices,
        batch_converter,
        model,
        device,
        alphabet
    )

Model Architecture
-------------------------------------------------------------------------------------------

SPURS Models
~~~~~~~~~~~

.. automodule:: spurs.models.stability.spurs
   :members: SPURS, SPURSConfig
   :special-members: __init__
   :show-inheritance:

ProteinMPNN Model
~~~~~~~~~~~~~~~~

.. automodule:: spurs.models.stability.mpnn
   :members: MPNN
   :special-members: __init__
   :show-inheritance:

ESM Model
~~~~~~~~~

.. automodule:: spurs.models.stability.esm
   :members: ESM
   :special-members: __init__
   :show-inheritance:

Transfer Model
~~~~~~~~~~~~~

.. automodule:: spurs.models.stability.org_transfer_model
   :members: TransferModel, TransferModelConfig
   :exclude-members: parse_PDB_biounits, parse_PDB, alt_parse_PDB_biounits, alt_parse_PDB, tied_featurize, loss_nll, loss_smoothed, StructureDataset, StructureDatasetPDB, StructureLoader, EncLayer, DecLayer, PositionWiseFeedForward, PositionalEncodings, CA_ProteinFeatures, ProteinFeatures, ProteinMPNN, get_protein_mpnn, LightAttention, gather_edges, gather_nodes, gather_nodes_t, cat_neighbors_nodes
   :special-members: __init__
   :show-inheritance:


Data Modules
-----------

.. automodule:: spurs.datamodules.datasets.megascale
   :members: MegaScaleDataset, MegaScaleTestDatasets
   :special-members: __init__
   :show-inheritance:

.. automodule:: spurs.datamodules.datasets.megascale_multi
   :members:
   :special-members: __init__
   :show-inheritance:

.. automodule:: spurs.datamodules.datasets.domainome
   :members: domainome
   :special-members: __init__
   :show-inheritance:

.. automodule:: spurs.datamodules.datasets.ddgbench
   :members: ddgBenchDataset
   :special-members: __init__
   :show-inheritance:

.. automodule:: spurs.datamodules.datasets.ddggeo
   :members: ddgGeo
   :special-members: __init__
   :show-inheritance:

.. automodule:: spurs.datamodules.datasets.fireport
   :members: FireProtDataset
   :special-members: __init__
   :show-inheritance:

.. note::
   - ddgBenchDataset is used for ssym-dir, ssym-inv, and S669 datasets
   - ddgGeo is used for S461, S783, S8754, S2648, S571, and S4346 datasets
