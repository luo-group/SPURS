# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'SPURS'
copyright = '2025, Ziang Li'
author = 'Ziang Li'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

extensions = [
    'sphinx.ext.autodoc',        # Automatically document Python modules
    'sphinx.ext.napoleon',       # Support for Google-style docstrings
    'sphinx.ext.viewcode',       # Add links to source code
    'sphinx.ext.intersphinx',    # Link to other projects' documentation
    'sphinx.ext.mathjax',        # Render math equations
    'sphinx.ext.githubpages',    # Enable GitHub Pages
]

# Mock imports for modules that are not installed
autodoc_mock_imports = [
<<<<<<< HEAD
    "torchvision",
    "torchmetrics",
    "torchtext",
    "torch_geometric",
    "sklearn",
    "scipy",
    "pytorch_lightning",
    "fairscale",
=======
    'torch',
    'pytorch_lightning',
    'hydra',
    'biopython',
    'einops',
    'fairscale',
    'e3nn',
    'spurs',
    'numpy',
    'pandas',
>>>>>>> 3b21f9ea77675ef6c7cc0e632c77811c5ce2c358
]

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

templates_path = ['_templates']
exclude_patterns = []

language = 'en'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

import sphinx_rtd_theme
extensions.append('sphinx_rtd_theme')
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Theme options
html_theme_options = {
    'navigation_depth': 4,
    'titles_only': False
}
