"""Setup configuration for trino-query-predictor package."""
from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open(os.path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='trino-query-predictor',
    version='1.0.0',
    author='Interactive Analytics',
    author_email='ia-staff@tmp.com',
    description='A production-ready REST API service that classifies Trino SQL queries',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://git.soma.tmp.com/sfdc-presto/trino-query-predictor/',
    packages=find_packages(exclude=['tests', 'tests.*']),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
    ],
    python_requires='>=3.8',
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'trino-query-predictor=query_predictor.service.app:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

