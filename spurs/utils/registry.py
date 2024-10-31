# https://github.com/BytedProtein/ByProt/blob/dd279dc85f76ee2c28c819b71bf3911b90159f0a/src/byprot/utils/registry.py
from spurs.datamodules import DATAMODULE_REGISTRY
from spurs.models import MODEL_REGISTRY
from spurs.tasks import TASK_REGISTRY

registry_dict = dict(
    datamodule=DATAMODULE_REGISTRY,
    task=TASK_REGISTRY,
    model=MODEL_REGISTRY
)

def get_module(group_name, module_name):
    group = registry_dict.get(group_name, None)
    if group is None:
        raise KeyError(f'{group_name} is not a valid registry group {registry_dict.keys()}.')
    
    return group.get(module_name)

def get_registered_modules(group_name):
    group = registry_dict.get(group_name)
    if group is not None:
        return group.keys()
    else:
        raise KeyError(f'{group_name} is not a valid registry group {registry_dict.keys()}.')

__all__ = [
    'get_module',
    'get_registered_modules'
]