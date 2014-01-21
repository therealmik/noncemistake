#!/usr/bin/python3

import gzip
import numpy
from data import from_ascii
import os

def from_file(fobj):
	"""Return a function that scores based on frequency counts read
	   from fobj.  If fobj is a str, it will open it for reading
	   first"""

	freqtable = numpy.bincount(
		numpy.memmap(fobj, mode="r"),
		minlength=256
	)

	def get_freq_score(data):
		"""Return a positive integer where the higher the
		   integer, the more 'like' the file the passed data
		   is"""
		score = 0
		if isinstance(data, str):
			data = numpy.fromstring(data, dtype=numpy.uint8)

		for d in data:
			score += freqtable[d]
		return score
	
	return get_freq_score

def from_fortunes():
	freqtable = numpy.zeros(256, dtype=numpy.uint32)

	if os.path.exists("/usr/share/games/fortunes/fortunes"):
		with open("/usr/share/games/fortunes/fortunes", "r") as fd:
			for line in fd:
				line = line.strip()
				if line == '%':
					continue
				freqtable += numpy.bincount(from_ascii(line), minlength=256)
	elif os.path.exists("/usr/share/games/fortune") and os.path.isdir("/usr/share/games/fortune"):
		for filename in os.listdir("/usr/share/games/fortune"):
			if filename.endswith(".dat") or filename.endswith(".u8"):
				continue
			fullpath = os.path.join("/usr/share/games/fortune", filename)
			with open(fullpath, "r") as fd:
				for line in fd:
					line = line.strip()
					if line == '%':
						continue
					freqtable += numpy.bincount(from_ascii(line), minlength=256)
	else:
		return num_obvious_chars

	def get_freq_score(data):
		"""Return a positive integer where the higher the
		   integer, the more 'like' the file the passed data
		   is"""
		score = 0
		if isinstance(data, str):
			data = numpy.fromstring(data, dtype=numpy.uint8)

		for d in data:
			score += freqtable[d]
		return score
	
	return get_freq_score

def num_obvious_chars(data):
	"""Return a positive integer representing the number of
	   'obvious' characters are in data.  By obvious, I mean
	   lower-case letters and spaces.  This is good for text
	   messages and chats, and ok for most english"""
	score = 0
	if isinstance(data, str):
		data = numpy.fromstring(data, dtype=numpy.uint8)

	for c in data:
		if c == 32:
			score += 1
		if c >= 97 and c <= 122:
			score += 1
	return score

