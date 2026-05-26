Installation
============

Prerequisites
------------

- CUDA-compatible GPU (recommended)
- Conda package manager

Installation Steps
----------------

1. Inference environment (recommended):

.. code-block:: bash

   conda create -n spurs_higher python=3.11 pip
   conda activate spurs_higher
   pip install -r requirements.inference.txt
   pip install -e .

2. Legacy training environment (optional):

.. code-block:: bash

   conda create -n spurs python=3.7 pip
   conda activate spurs
   pip install -r requirements.training-legacy.txt
   pip install -e .
   pip install git+https://github.com/facebookresearch/esm.git

Verification
-----------

To verify your installation, you can run a simple test:

.. code-block:: python

   from spurs.inference import get_SPURS, parse_pdb
   model, cfg = get_SPURS_from_hub()
