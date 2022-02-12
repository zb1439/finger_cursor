import os
import shutil
from setuptools import find_packages, setup

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(CUR_PATH, 'build')
if os.path.isdir(path):
    print('delete ', path)
    shutil.rmtree(path)
path = os.path.join(CUR_PATH, 'dist')
if os.path.isdir(path):
    print('delete ', path)
    shutil.rmtree(path)

head = "#!/bin/bash\n\n"
with open("tools/finger_cursor", "w") as f:
    f.write(head + f"python3 {os.path.join(CUR_PATH, 'tools/launch.py')} $@")

setup(
    name='finger_cursor',
    author='Zhibo Fan',
    description="Mouse-free cursor controller based on any camera",
    version='0.1',
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        'opencv-python>=4.5.3',
        'numpy>=1.20',
        'matplotlib',
        'tqdm>4.29.0',
        'sklearn',
        'easydict',
        'colorama',
        'tabulate',
        'pynput==1.7.3',
        'pyobjc==7.3',
        'mediapipe',
        'pygame==2.1.0',
        'nes-py',
        "scikit-image",
        "keyboard"
    ],
    scripts=["tools/finger_cursor"],
)
