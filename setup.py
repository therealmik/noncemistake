#!/usr/bin/python

from distutils.core import setup

setup(	name='NonceMistake',
	version='1.0',
	description='Break AES-CTR interactively',
	license='MIT',
	author='Michael Samuel',
	author_email='mik@miknet.net',
	url='https://miknet.net/',
	packages=['mikcryptoolkit'],
	scripts=['noncemistake'],
	requires=['numpy', 'Crypto', 'pygame'],
)
