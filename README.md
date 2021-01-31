# asm2vec-pytorch

Unofficial implementation of `asm2vec` using pytorch ( with GPU acceleration )  
The details of the model can be found in the original paper: [(sp'19) Asm2Vec: Boosting Static Representation Robustness for Binary Clone Search against Code Obfuscation and Compiler Optimization](https://www.computer.org/csdl/proceedings-article/sp/2019/666000a038/19skfc3ZfKo)  

## Requirements

This implementation is written in python 3.8.

You need `r2pipe` to run `bin2asm.py`  
You need `click` to run both `bin2asm.py` and `train.py`  
You need `torch` to run `train.py` and `asm2vec` library

## Benchmark

An implementation already exists here: [Lancern/asm2vec](https://github.com/Lancern/asm2vec)  
Following is the benchmark of training 1000 functions in 1 epoch.

| Implementation | Time (s) |
| :-: | :-: |
| [Lancern/asm2vec](https://github.com/Lancern/asm2vec) | 202.23 |
| [oalieno/asm2vec-pytorch](https://github.com/oalieno/asm2vec-pytorch) (with CPU) | 9.11 |
| [oalieno/asm2vec-pytorch](https://github.com/oalieno/asm2vec-pytorch) (with GPU) | 0.97 |

## Usage

```
Usage: bin2asm.py [OPTIONS]

  Extract assembly functions from binary executable

Options:
  -i, --input TEXT   input directory / file  [required]
  -o, --output TEXT  output directory
  -l, --len INTEGER  ignore assembly code with instructions amount smaller
                     than minlen

  --help             Show this message and exit.
```

```bash
# Example
python bin2asm.py -i /bin/ -o asm/
```

---

```
Usage: train.py [OPTIONS]

Options:
  -i, --input TEXT     training data folder  [required]
  -o, --output TEXT    output model path
  -e, --epoch INTEGER  training epoch
  --help               Show this message and exit.
```

```bash
# Example
python train.py -i asm/ -o model.pt --epoch 100
```
