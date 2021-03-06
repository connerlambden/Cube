from setuptools import setup

setup(
    name='sc_cube',
    version='1.0',    
    description='Simple Nonparametric Gene-Gene Relationship Search Algorithm',
    url='https://github.com/connerlambden/Cube',
    author='Conner Lambden',
    author_email='conner@connerpro.com',
    license='MIT',
    packages=['sc_cube'],
    install_requires=['numpy', 
                      'jit',
                      'networkx',
                      'scipy',
                      'pandas'               
                      ],

    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
)
