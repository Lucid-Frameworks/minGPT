from setuptools import setup

setup(name='tabGPT',
      version='0.0.1',
      author='Felix Wick',
      packages=['tabgpt'],
      description="An adaption of Andrej Karpathy's minGPT for tabular data",
      license='MIT',
      install_requires=[
            'torch',
            'transformers',
            'pandas',
      ],
)
