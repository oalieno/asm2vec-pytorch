# asm2vec-pytorch

<a><img alt="release 1.0.3" src="https://img.shields.io/badge/release-v1.0.0-yellow?style=for-the-badge"></a>
<a><img alt="mit" src="https://img.shields.io/badge/license-MIT-brightgreen?style=for-the-badge"></a>
<a><img alt="python" src="https://img.shields.io/badge/-python-9cf?style=for-the-badge&logo=python"></a>

Unofficial implementation of `asm2vec` using pytorch ( with GPU acceleration )  
The details of the model can be found in the original paper: [(sp'19) Asm2Vec: Boosting Static Representation Robustness for Binary Clone Search against Code Obfuscation and Compiler Optimization](https://www.computer.org/csdl/proceedings-article/sp/2019/666000a038/19skfc3ZfKo)  

## Requirements

* python >= 3.10
* radare2
* Packages listed in `requirements.txt`

## Install

```
pip install -r requirements.txt && 
python setup.py install
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

### TODO - update this with description about to how use etc

## Tests

### Run test suite

* Run all tests: ``python -m unittest discover -v``
* Run a certain module's tests: ``python -m unittest -v test.test_binary_to_asm``
* Run a certain test class: ``python -m unittest -v test.test_binary_to_asm.TestBinaryToAsm``
* Run a certain test method: 

  ``python -m unittest -v test.test_binary_to_asm.TestBinaryToAsm.test_sha3``

### Coverage

* Create report: ``coverage run -m unittest discover -v``
* Read report: ``coverage report -m``