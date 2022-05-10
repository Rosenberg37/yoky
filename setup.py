from distutils.core import setup

import pkg_resources
from setuptools import find_packages

setup(
    name='yoky',  # 包名
    version='0.0.1',  # 版本号
    description='A very simple framework for Deep Learning',
    long_description=open('README.md').read(),
    author='Rosenberg',
    author_email='986301306@qq.com',
    maintainer='Rosenberg',
    maintainer_email='986301306@qq.com',
    license='MIT License',
    packages=find_packages(),
    url='https://github.com/Rosenberg37/yoky',
    platforms=["all"],
    classifiers=[],
)
