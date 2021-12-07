import os
from setuptools import setup, find_packages
from distutils.core import setup

# Utility function to read the README file.

def read(fname):
	return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
	name = "corpus_similarity",
	version = "1.0.12",
	author = "Jonathan Dunn, Haipeng Li",
	author_email = "jonathan.dunn@canterbury.ac.nz",
	description = ("Measuring corpus similarity in Python"),
	license = "GNU GENERAL PUBLIC LICENSE v3",
	url = "https://github.com/jonathandunn/corpus_similarity",
	keywords = "text analytics, natural language processing, computational linguistics, corpus, corpora, similarity",
	packages = find_packages(exclude=["*.pyc", "__pycache__"]),
	package_data={'': ['corpus_similarity.in_domain_features.*',
						'corpus_similarity.out_of_domain_features.*',
						'corpus_similarity.threshold_values.*',
						'corpus_similarity.scaler_values.*']},
	install_requires=["cytoolz",
						"numpy",
						"clean-text",
						"scipy",
						"sklearn",
					  	"spacy",

						],
	include_package_data=True,
	long_description=read('README.md'),
    long_description_content_type='text/markdown',
	)