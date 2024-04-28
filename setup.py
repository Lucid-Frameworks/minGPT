from setuptools import setup

setup(name='tinyGPT',
      version='0.0.1',
      # author='Andrej Karpathy',
      author='Ziliang Peng',
      packages=['tinygpt'],
      description='A tinygrad port of Andrej Karpathy\'s minGPT',
      license='MIT',
      install_requires=[
            'torch',
            'tinygrad',
      ],
)
