#!/usr/bin/python3

from __future__ import unicode_literals, division

import numpy
import os
import binascii
import sys

if sys.version_info.major == 2:
	str = unicode

def from_hex(x):
	"""Convert hex string into a numpy.uint8 array"""
	return numpy.fromstring(
		binascii.a2b_hex(x),
		dtype=numpy.uint8
	)

def to_hex(a):
	"""Convert numpy.dtype into hex string"""
	return str(
		binascii.b2a_hex(
			a.tostring()
		),
		"UTF-8"
	)

def from_base64(x):
	"""Convert base64 string into a numpy.uint8 array"""
	return numpy.fromstring(
		binascii.a2b_base64(x),
		dtype=numpy.uint8
	)

def to_base64(a):
	"""Convert numpy.dtype into base64 string"""
	return str(
		binascii.b2a_base64(
			a.tostring()
		).strip(),
		"UTF-8"
	)

def to_ascii(a):
	"""Convert a numpy.uint8 array into a string"""
	return str(
		a.tostring(),
		"UTF-8",
		errors='ignore'
	)
		
def from_ascii(s):
	"""Convert a numpy.uint8 array into a string"""
	return numpy.array(
		[
			ord(c)
			for c in s
		],
		dtype=numpy.uint8
	)

def random_bytes(num):
	"""Return num random bytes (using os.urandom())"""
	return numpy.fromstring(
		os.urandom(num),
		dtype=numpy.uint8
	)

def printable(a):
	"""Convert a into ascii, quoting non-printable characters"""
	return "".join([
		chr(c).isprintable() and chr(c) or "\\x{0:02x}".format(c)
		for c in a
	])

def dice(data, size):
	"""Basically reshapes the passed data into blocks of size"""
	return data.reshape((data.shape[0] // size, size))

undice = numpy.concatenate

def slice(data, size):
	"""Pretend ciphertext is a 2d bitmap, and slice column-wise"""
	return dice(data, size).T

def unslice(data):
	"""Reverse of slice"""
	return numpy.concatenate(data.T)

