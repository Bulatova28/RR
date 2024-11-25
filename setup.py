from setuptools import setup, find_packages

setup(
    name='RR_PROJECT_BULATOVA_POPP', 
    version='0.1',  
    description='Package for visualization of statistical tests',  
    long_description=open('README.md').read(),  
    long_description_content_type='text/markdown', 
    author='Bulatova Viktoriia, Popp Sofia',  
    url='https://github.com/Bulatova28/RR_project_Bulatova_Popp',  
    packages=find_packages(), 
    install_requires=[  
        'numpy',  
        'pandas',  
        'scipy',  
        'seaborn', 
        'matplotlib',  
        'joypy',  
        'typing'
    ],
    classifiers=[  
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research', 
        'Topic :: Scientific/Engineering :: Visualization/Statistics'
    ],
)
