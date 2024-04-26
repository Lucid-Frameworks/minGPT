from setuptools import setup

setup(name='tinyGPT',
      version='0.0.1',
      author='Andrej Karpathy',
      packages=['tinygpt'],
      description='A PyTorch re-implementation of GPT',
      license='MIT',
      install_requires=[
            'torch',
            'tinygrad',
      ],
)
