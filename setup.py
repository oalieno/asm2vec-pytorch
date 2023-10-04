import os
from setuptools import setup, find_packages
from setuptools.command.install import install as _install

from asm2vec.version import VERSION


class install(_install):
    @staticmethod
    def _setup_radare2() -> None:
        if os.system('r2env shell "r2 -v"') == 0:
            print("radar2 already set up!")
            return
        os.system("r2env init")
        os.system("r2env add radare2")
        os.system("r2env use radare2@git")

    def run(self):
        _install.run(self)
        self._setup_radare2()


def readme():
    with open('README.md') as f:
        return f.read()


def read_requirements():
    with open('requirements.txt') as f:
        return [s for s in f.read().split('\n') if not ('--index-url' in s)]


setup(
    name='asm2vec',
    version=VERSION,
    description="Jamf's implementation of asm2vec using pytorch",
    long_description=readme(),
    author='oalieno/jamf',
    author_email='jamie.nutter@jamf.com',
    license='MIT License',
    install_requires=read_requirements(),
    packages=find_packages(),
    zip_safe=False,
    include_package_data=True,
    test_suite='nose.collector',
    tests_require=['nose'],
    cmdclass={'install': install}
)
