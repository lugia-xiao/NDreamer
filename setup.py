from setuptools import setup, find_packages

setup(
    name='ndreamer',
    version='1.0.0',
    author='Xiao Xiao',
    author_email='xiao.xiao.xx244@yale.edu',
    packages=find_packages(),
    install_requires=[
        'torch',
        'scanpy',
        'numpy',
        'tqdm',
        'umap-learn',
        'scipy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'anndata',
        'scib'
    ],
    url='http://pypi.python.org/pypi/ndreamer/',
    license='LICENSE.txt',
    description='Statistically decomposing condition-related signals, effect modifiers, and measurement errors of complex forms in scRNA-seq with neural discrete representation learning',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown'
)