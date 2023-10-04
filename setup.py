import os
import sys
from setuptools import setup, find_packages
from setuptools.command.install import install as _install

from asm2vec.version import VERSION, radare2_version


class install(_install):
    @staticmethod
    def _setup_radare2() -> None:
        if sys.platform.startswith("linux"):
            os.system("apt-get update")
            os.system("apt-get install -y --no-install-recommends wget")
            os.system(f"wget -O /tmp/radare2_${radare2_version}_arm64.deb https://github.com/radareorg/radare2/releases/download/${radare2_version}/radare2_${radare2_version}_arm64.deb")
            os.system(f"dpkg -i /tmp/radare2_${radare2_version}_arm64.deb")
            os.system("r2pm init")
            os.system("r2pm update")
            os.system(f"rm /tmp/radare2_${radare2_version}_arm64.deb")
        elif sys.platform.startswith("darwin"):
            os.system("brew install radare2")
        else:
            print("Ensure 'radar2' is installed...")

    def run(self):
        self._setup_radare2()
        _install.run(self)



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
