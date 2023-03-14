
from setuptools import setup, find_packages

#ToDO, make nicer, for example by importing requirements or sth
setup(
    name='epymarl',
    version='0.0.1',
    #install_requires=['numpy', 'gym', 'torch', 'tensorboard', 'matplotlib', 'pandas', 'scipy', 'tqdm', 'seaborn', 'pyyaml', 'cloudpickle', 'psutil', 'pyglet', 'pybullet', 'gym[box2d]', 'gym[all]', 'gym[robotics]', 'gym[classic_control]', 'gym[atari]'],
    packages=find_packages()
    )

