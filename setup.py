from setuptools import setup
import sys

assert sys.version_info.major == 3 and sys.version_info.minor >= 7, \
    "LevelUp is designed to work with Python 3.7 and greater." \
    + "Please use a compatible version."

setup(
    name='levelup',
    py_modules=['levelup'],
    install_requires=[
        'gym[atari,box2d,classic_control]',
        'numpy==1.19.2',
        'torch==1.6.0',
        'gym==0.17.3',
        'jupyterlab==2.2.9',
        'pandas==1.1.4',
        'plotly=4.13.0'
    ],
    description="Implementations of well known reinforcement learning algorithms.",
    author="Mickey Beurskens",
)
