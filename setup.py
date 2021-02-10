from setuptools import setup, find_packages

setup(
    name='asm2vec',
    version='1.0.0',
    description='Unofficial implementation of asm2vec using pytorch',
    install_requires=['torch>=1.7,<2'
                      'click>=7.1,<8'
                      'r2pipe>=1.5,<2'],
    author='oalieno',
    author_email='jeffrey6910@gmail.com',
    license='MIT License',
    packages = find_packages(),
)
