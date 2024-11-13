import setuptools

NAME = 'ptgctl'

deps = {
        'image': ['Pillow', 'opencv-python', 'numpy', 'supervision'],
    'audio': ['sounddevice', 'soundfile'],
}

import os, imp
version = imp.load_source('ptgctl.__version__', os.path.abspath(os.path.join(__file__, '../ptgctl/__version__.py'))).__version__

setuptools.setup(
    name=NAME,
    version=version,
    description='A Python Library and Command Line tool for the PTG API',
    long_description=open('README.md').read().strip(),
    long_description_content_type='text/markdown',
    author='Bea Steers',
    author_email='bea.steers@gmail.com',
    url=f'https://github.com/VIDA-NYU/{NAME}',
    packages=setuptools.find_packages(),
    entry_points={'console_scripts': ['{name}={name}:main'.format(name=NAME)]},
    install_requires=[
        'requests', 'websockets>=14.0.0', 'fire>=0.5.0',
        # 'fire @ git+ssh://git@github.com/google/python-fire@master#egg=fire', 
        'tabulate', 'tqdm', 'IPython',
        'redis_record>=0.0.4',
    ],
    extras_require={
        'test': ['pytest', 'pytest-cov'],
        'doc': ['sphinx-rtd-theme'],
        **deps,
        'all': {vi for v in deps.values() for vi in v},
    },
    license='MIT License',
    keywords='ptg api cli server data streams')
