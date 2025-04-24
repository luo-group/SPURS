Installation
============

Prerequisites
------------

- CUDA-compatible GPU (recommended)
- Conda package manager

Installation Steps
----------------

1. Create a new conda environment:

.. code-block:: bash

   conda create -n spurs python=3.7 pip
   conda activate spurs

2. Install SPURS and its dependencies:

.. code-block:: bash

   pip install -e .
   pip install git+https://github.com/facebookresearch/esm.git
   pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113

Verification
-----------

To verify your installation, you can run a simple test:

.. code-block:: python

   from spurs.inference import get_SPURS, parse_pdb
   model, cfg = get_SPURS_from_hub()

