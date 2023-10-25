import os

__home__ = os.path.dirname(os.path.abspath(__path__[0]))
__data__ = os.path.join(__home__, "data")

__all__ = [
    "__data__", "__home__", "binary_to_asm", "data", "datatype", "model", "similarity", "tensors", "test", "train",
    "utilities", "version"
]
