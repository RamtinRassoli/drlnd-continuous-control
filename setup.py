#!/usr/bin/env python

from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(name='unity_banana_navigation',
      version='0.1.0',
      description='Unity Banana Navigation',
      packages=find_packages(),
      install_requires=required,
      long_description="An implementation of a Double Deep Q-network agent playing the Unity Banana Navigation game."
      )
