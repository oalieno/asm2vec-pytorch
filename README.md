# asm2vec-pytorch

Unofficial implementation of `asm2vec` using pytorch ( with GPU acceleration )  
The details of the model can be found in the original paper: [(sp'19) Asm2Vec: Boosting Static Representation Robustness for Binary Clone Search against Code Obfuscation and Compiler Optimization](https://www.computer.org/csdl/proceedings-article/sp/2019/666000a038/19skfc3ZfKo)  

## Requirements

This implementation is written in python 3.8.

You need `r2pipe` to run `bin2asm.py`  
You need `click` to run `bin2asm.py`, `train.py` and `test.py`  
You need `torch` to run `train.py`, `test.py` and `asm2vec` library

## Benchmark

An implementation already exists here: [Lancern/asm2vec](https://github.com/Lancern/asm2vec)  
Following is the benchmark of training 1000 functions in 1 epoch.

| Implementation | Time (s) |
| :-: | :-: |
| [Lancern/asm2vec](https://github.com/Lancern/asm2vec) | 202.23 |
| [oalieno/asm2vec-pytorch](https://github.com/oalieno/asm2vec-pytorch) (with CPU) | 9.11 |
| [oalieno/asm2vec-pytorch](https://github.com/oalieno/asm2vec-pytorch) (with GPU) | 0.97 |

## Usage

### bin2asm.py

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

### train.py

```
Usage: train.py [OPTIONS]

Options:
  -i, --input TEXT                training data folder  [required]
  -o, --output TEXT               output model path  [default: ./model.pt]
  -d, --ebedding-dimension INTEGER
                                  embedding dimension  [default: 100]
  -b, --batch-size INTEGER        batch size  [default: 1024]
  -e, --epochs INTEGER            training epochs  [default: 10]
  -n, --neg-sample-num INTEGER    negative sampling amount  [default: 25]
  -c, --device TEXT               hardware device to be used: cpu / cuda /
                                  auto  [default: auto]

  --help                          Show this message and exit.
```

```bash
# Example
python train.py -i asm/ -o model.pt --epoch 100
```

### test.py

```
Usage: test.py [OPTIONS]

Options:
  -i, --input TEXT      target function  [required]
  -m, --model TEXT      model path  [required]
  -e, --epochs INTEGER  training epochs  [default: 10]
  -l, --limit INTEGER   limit the amount of output probability result
  -c, --device TEXT     hardware device to be used: cpu / cuda / auto
                        [default: auto]

  -p, --pretty          pretty print table  [default: False]
  --help                Show this message and exit.
```

```bash
# Example
python test.py -i asm/123456 -m model.pt
```

```
┌──────────────────────────────────────────┐
│    endbr64                               │
│  ➔ push r15                              │
│    push r14                              │
├────────┬─────────────────────────────────┤
│ 34.68% │ [rdx + rsi*CONST + CONST]       │
│ 20.29% │ push                            │
│ 16.22% │ r15                             │
│ 04.36% │ r14                             │
│ 03.55% │ r11d                            │
└────────┴─────────────────────────────────┘
```
