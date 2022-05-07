# asm2vec-pytorch

<a><img alt="release 1.0.0" src="https://img.shields.io/badge/release-v1.0.0-yellow?style=for-the-badge"></a>
<a><img alt="mit" src="https://img.shields.io/badge/license-MIT-brightgreen?style=for-the-badge"></a>
<a><img alt="python" src="https://img.shields.io/badge/-python-9cf?style=for-the-badge&logo=python"></a>

Unofficial implementation of `asm2vec` using pytorch ( with GPU acceleration )  
The details of the model can be found in the original paper: [(sp'19) Asm2Vec: Boosting Static Representation Robustness for Binary Clone Search against Code Obfuscation and Compiler Optimization](https://www.computer.org/csdl/proceedings-article/sp/2019/666000a038/19skfc3ZfKo)  

## Requirements

python >= 3.6

| packages | for |
| --- | --- |
| r2pipe | `scripts/bin2asm.py` |
| click | `scripts/*` |
| torch | almost all code need it |

You also need to install `radare2` to run `scripts/bin2asm.py`. `r2pipe` is just the python interface to `radare2`

If you only want to use the library code, you just need to install `torch`

## Install

```
python setup.py install
```

or

```
pip install git+https://github.com/oalieno/asm2vec-pytorch.git
```

## Benchmark

An implementation already exists here: [Lancern/asm2vec](https://github.com/Lancern/asm2vec)  
Following is the benchmark of training 1000 functions in 1 epoch.

| Implementation | Time (s) |
| :-: | :-: |
| [Lancern/asm2vec](https://github.com/Lancern/asm2vec) | 202.23 |
| [oalieno/asm2vec-pytorch](https://github.com/oalieno/asm2vec-pytorch) (with CPU) | 9.11 |
| [oalieno/asm2vec-pytorch](https://github.com/oalieno/asm2vec-pytorch) (with GPU) | 0.97 |

## Get Started

```bash
python scripts/bin2asm.py -i /bin/ -o asm/
```

First generate asm files from binarys under `/bin/`.  
You can hit `Ctrl+C` anytime when there is enough data.

```bash
python scripts/train.py -i asm/ -l 100 -o model.pt --epochs 100
```

Try to train the model using only 100 functions and 100 epochs for a taste.  
Then you can use more data if you want.

```bash
python scripts/test.py -i asm/123456 -m model.pt
```

After you train your model, try to grab an assembly function and see the result.  
This script will show you how the model perform.  
Once you satisfied, you can take out the embedding vector of the function and do whatever you want with it.

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
  -o, --output TEXT               output model path  [default: model.pt]
  -m, --model TEXT                load previous trained model path
  -l, --limit INTEGER             limit the number of functions to be loaded
  -d, --ebedding-dimension INTEGER
                                  embedding dimension  [default: 100]
  -b, --batch-size INTEGER        batch size  [default: 1024]
  -e, --epochs INTEGER            training epochs  [default: 10]
  -n, --neg-sample-num INTEGER    negative sampling amount  [default: 25]
  -a, --calculate-accuracy        whether calculate accuracy ( will be
                                  significantly slower )

  -c, --device TEXT               hardware device to be used: cpu / cuda /
                                  auto  [default: auto]

  -lr, --learning-rate FLOAT      learning rate  [default: 0.02]
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
  -i, --input TEXT              target function  [required]
  -m, --model TEXT              model path  [required]
  -e, --epochs INTEGER          training epochs  [default: 10]
  -n, --neg-sample-num INTEGER  negative sampling amount  [default: 25]
  -l, --limit INTEGER           limit the amount of output probability result
  -c, --device TEXT             hardware device to be used: cpu / cuda / auto
                                [default: auto]

  -lr, --learning-rate FLOAT    learning rate  [default: 0.02]
  -p, --pretty                  pretty print table  [default: False]
  --help                        Show this message and exit.
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

### compare.py

```
Usage: compare.py [OPTIONS]

Options:
  -i1, --input1 TEXT          target function 1  [required]
  -i2, --input2 TEXT          target function 2  [required]
  -m, --model TEXT            model path  [required]
  -e, --epochs INTEGER        training epochs  [default: 10]
  -c, --device TEXT           hardware device to be used: cpu / cuda / auto
                              [default: auto]

  -lr, --learning-rate FLOAT  learning rate  [default: 0.02]
  --help                      Show this message and exit.
```

```bash
# Example
python compare.py -i1 asm/123456 -i2 asm/654321 -m model.pt -e 30
```

```
cosine similarity : 0.873684
```
