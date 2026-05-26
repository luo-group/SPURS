# some code adated from https://github.com/BytedProtein/ByProt and https://github.com/ashleve/lightning-hydra-template
"""Top-level SPURS package with lazy submodule loading.

This keeps ``import spurs`` lightweight for inference-only use cases.
"""

import importlib

__all__ = ["models", "tasks", "utils", "datamodules"]


def __getattr__(name):
    if name in __all__:
        module = importlib.import_module(f"spurs.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module 'spurs' has no attribute '{name}'")
