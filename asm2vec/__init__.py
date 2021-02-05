import importlib

__all__ = ['model', 'datatype', 'utils']

for module in __all__:
    importlib.import_module(f'.{module}', 'asm2vec')
