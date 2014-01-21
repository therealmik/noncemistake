#!/usr/bin/python3

from Crypto.Cipher import AES
import numpy
import sys
from .data import *

def pkcs7_unpad(s, blocksize=16):
	"""Remove PKCS#7 padding bytes"""
	numbytes = s[-1]
	if numbytes > blocksize:
		raise ValueError("Invalid padding amount: " + str(numbytes))
	for i in range(numbytes):
		if s[-1 - i] != numbytes:
			raise ValueError("Invalid padding bytes")
	return s[:s.size-numbytes]

def pkcs7_pad(s, blocksize=16):
	"""Add PKCS#7 padding.	Add a whole block if over"""
	num_padding_bytes = blocksize - (s.size % blocksize)
	pad = numpy.repeat(numpy.uint8(num_padding_bytes), num_padding_bytes)
	return numpy.concatenate([s, pad])

def cbc_encrypt(s, key, iv, cipher=AES):
	"""Encrypt s using the secret key and the public iv
	   with the CBC block mode and a Crypto.Cipher (AES
	   by default)"""
	assert(iv.size == cipher.block_size)

	s = pkcs7_pad(s, cipher.block_size)
	ctx = cipher.new(key.tostring(), mode=cipher.MODE_ECB)
	prev = iv
	ciphertext = []
	for block in dice(s, cipher.block_size):
		prev = numpy.fromstring(
			ctx.encrypt(
				(prev ^ block).tostring()
			),
			dtype=numpy.uint8
		)
		ciphertext.append(prev)
	return undice(ciphertext)

def cbc_decrypt(ciphertext, key, iv, cipher=AES):
	assert(iv.size == cipher.block_size)
	ctx = cipher.new(key.tostring(), mode=cipher.MODE_ECB)
	prev = iv
	plaintext = []
	for block in dice(ciphertext, cipher.block_size):
		plaintext.append(
			numpy.fromstring(
				ctx.decrypt(block.tostring()),
				dtype=numpy.uint8
			) ^ prev
		)
		prev = block
	return pkcs7_unpad(undice(plaintext))

def ecb_encrypt(plaintext, key, _iv=None, cipher=AES):
	ctx = cipher.new(key, mode=cipher.MODE_ECB)
	return numpy.fromstring(
		ctx.encrypt(
			pkcs7_pad(plaintext).tostring()
		),
		dtype=numpy.uint8
	)

def ecb_decrypt(ciphertext, key, _iv=None, cipher=AES):
	ctx = cipher.new(key.tostring(), mode=cipher.MODE_ECB)
	return pkcs7_unpad(
		numpy.fromstring(
			ctx.decrypt(ciphertext),
			dtype=numpy.uint8
		)
	)

def test_for_ecb(ciphertext, blocksize=16):
	for i in range(len(ciphertext) - blocksize - blocksize):
		block = ciphertext[i:i+blocksize]
		for j in range(i+blocksize, len(ciphertext) - blocksize, blocksize):
			if not (block ^ ciphertext[j:j+blocksize]).any():
				return True
	return False

def find_blocksize(oracle_func):
	size_test = from_ascii("")
	origSize = oracle_func(size_test).size
	testSize = origSize
	while origSize == testSize:
		size_test = numpy.concatenate([size_test, from_ascii("A")])
		testSize = oracle_func(size_test).size
	return testSize - origSize

def _ecb_bruteforce_block(expected, pad, blocksize, oracle_func):
	for i in (numpy.array([i], dtype=numpy.uint8) for i in range(256)):
		chosen_plaintext = numpy.concatenate([pad, i])
		ciphertext = oracle_func(chosen_plaintext)[:blocksize]
		if not (expected ^ ciphertext).any():
			return i
	raise ValueError("Unable to find next char - some assumption is wrong?")

def ecb_find_block(cache, blocksize, blocknum, pad, oracle_func):
	"""So we're going to push most of the previous block into the target
	   block, then test our first block (containing the same data + 1
	   random byte) into it"""
	found = from_ascii("")

	for i in range(blocksize):
		pad = pad[1:]
		orig_ciphertext = cache[blocksize-i-1][blocknum]
		try:
			nextChar = _ecb_bruteforce_block(orig_ciphertext, numpy.concatenate([pad, found]), blocksize, oracle_func)
			found = numpy.concatenate([found, nextChar])
		except ValueError:
			break
	return found

def ecb_create_ciphertext_cache(blocksize, oracle_func, extra_pad=0):
	"""This creates an array of blocksize x num_blocks ciphertexts, for
	   byte-at-a-time bruteforcing."""
	return [
		dice(
			oracle_func(numpy.repeat(numpy.uint8(65), i+extra_pad)),
			blocksize
		)
		for i in range(blocksize)
	]
	
def determine_prepend_size(oracle_func, blocksize):
	baseline = oracle_func(numpy.repeat(numpy.uint8(65), blocksize))

	def determine_prepend_whole_blocks():
		"""Found out how many whole blocks worth of prepend there is, returing the byte offset of the end"""
		comparison = oracle_func(numpy.repeat(numpy.uint8(66), blocksize))
		xored = baseline ^ comparison
		for i in range(0, xored.size, blocksize):
			if xored[i:i+blocksize].any():
				return (i // blocksize) # we subtract one, because we *didn't* match this time
		raise RuntimeError("Unable to determine prepend size - this can't be ECB encrypted, or prepend is dynamic")

	num_whole_blocks = determine_prepend_whole_blocks()

	# So we now know there's num_whole_blocks worth of prepend that we
	# don't control.  The question is, how many bytes worth of the next
	# block do we control?
	#
	# A way we can find out is to compare to our baseline of a whole
	# block worth of 'A', checking to see how many 'B' we can inject
	# before the next block is modified.

	def calc_test_slice():
		"""Return a slice of the ciphertext for the block after the
		   first one we control some of"""
		slice_start = (num_whole_blocks + 1) * blocksize
		slice_end = slice_start + blocksize
		return slice(slice_start, slice_end)
	test_slice = calc_test_slice()

	for i in range(1, blocksize+1):
		chosen_plaintext = numpy.concatenate([
			numpy.repeat(numpy.uint8(66), i),
			numpy.repeat(numpy.uint8(65), blocksize-i)
		])
		# See how many non-'A' we can slip in our non-matching block before we alter the next block
		comparison = oracle_func(chosen_plaintext) ^ baseline
		if comparison[test_slice].any():
			return ((num_whole_blocks+1) * blocksize) - i + 1

	# Well, if we're here then the padding must've been a multiple of 
	# the blocksize in length
	return (num_whole_blocks * blocksize)

def _advanced_ecb_bruteforce_block(expected, pad, blocksize, oracle_func, controlled_block):
	test_slice = slice(controlled_block*blocksize, (controlled_block+1)*blocksize)

	for i in (numpy.array([i], dtype=numpy.uint8) for i in range(256)):
		chosen_plaintext = numpy.concatenate([pad, i])
		ciphertext = oracle_func(chosen_plaintext)[test_slice]
		if not (expected ^ ciphertext).any():
			return i
	raise ValueError("Unable to find next char - some assumption is wrong?")

def advanced_ecb_find_block(cache, blocksize, blocknum, pad, oracle_func, num_extra_pad, controlled_block):
	"""So we're going to push most of the previous block into the target
	   block, then test our first block (containing the same data + 1
	   random byte) into it"""
	found = from_ascii("")

	# For challenge 14
	extra_pad = numpy.repeat(numpy.uint8(65), num_extra_pad)

	for i in range(blocksize):
		pad = pad[1:]
		orig_ciphertext = cache[blocksize-i-1][blocknum]
		try:
			nextChar = _advanced_ecb_bruteforce_block(
				orig_ciphertext,
				numpy.concatenate([extra_pad, pad, found]),
				blocksize,
				oracle_func,
				controlled_block
			)
			found = numpy.concatenate([found, nextChar])
		except ValueError:
			break
	return found

def change_cbc_byte(ciphertext, blocksize, bytenum, from_value, to_value):
	"""Change the value of a byte of CBC-encrypted ciphertext from it's
	   expected value to a new value.
	   This will destroy the block preceeding the mangled byte - as that's
	   what we're actually editing, and block ciphers are hopefully a
	   random permutation.
	   This won't propogate further than the intended block, because we're
	   not changing an other ciphertexts - CBC only XORs the preceeding
	   ciphertext against the decrypted plaintext."""
	xor_mask = numpy.zeros_like(ciphertext)
	xor_mask[bytenum-blocksize] = from_value ^ to_value
	return ciphertext ^ xor_mask

class _Found(Exception):
	def __init__(self, plaintext):
		self.plaintext = plaintext

def _cbc_padding_attack(block, oracle, sofar):
	"""Recursive function to perform CBC padding attack. All the
	   cool kids do recursion.  Oh, and this recovers in the case
	   of a false-positive on padding (eg. finding 0x02, 0x02 on
	   the first iteration)"""
	# Break out of recursion if we have a whole block
	if sofar.size == 16:
		raise _Found(sofar)

	this_size = numpy.uint8(sofar.size + 1)
	target_byte = 16 - this_size

	sofar ^= this_size
	testiv = numpy.concatenate([
		numpy.zeros(16-sofar.size, dtype=numpy.uint8),
		sofar
	])

	for i in numpy.arange(256, dtype=numpy.uint8):
		testiv[target_byte] = i
		if oracle(testiv, block):
			_cbc_padding_attack(block, oracle, numpy.concatenate([[i], sofar]) ^ this_size)

def cbc_padding_attack(iv, block, oracle):
	"""This attack uses CBC block corruption to try to make the
	   target block look like valid padding.
	   To convert that into the plain text you:
	   - xor against 0x10 (a full block of PKCS#7 padding
	   - xor against the IV/previous block

	   Much like length extension attacks, once you know how to do
	   this attack, you have trouble believing that you didn't know
	   about it all along."""
	try:
		_cbc_padding_attack(block, oracle, from_ascii(""))
		raise Exception("Unable to perform CBC padding attack")
	except _Found as f:
		return f.plaintext ^ iv
	
def _make_ctr_block_big(nonce, counter):
	return _make_ctr_block_little(nonce.byteswap(), counter.byteswap())

def _make_ctr_block_little(nonce, counter):
	return nonce.tostring() + counter.tostring()

if sys.byteorder == 'big':
	_make_ctr_block = _make_ctr_block_big
elif sys.byteorder == 'little':
	_make_ctr_block = _make_ctr_block_little
else:
	raise RuntimeError("Unknown byteorder")

def ctr_encrypt(s, key, nonce, cipher=AES):
	"""Encrypt s using the secret key and the public iv
	   with the CTR block mode and a Crypto.Cipher (AES
	   by default).  Nonce should be a numpy.uint64.
	   Only re-use a nonce for the same key if you're trying to
	   prove how insecure it is to do so.
	   No need to worry about s being more than 2^68 bytes long,
	   as that would require 2^69 bytes of memory to fit in the
	   address space of the process..."""
	nonce = numpy.uint64(nonce)

	ctx = cipher.new(key.tostring(), mode=cipher.MODE_ECB)
	num_blocks = (s.size + (cipher.block_size - 1)) // cipher.block_size

	keystream = undice([
		numpy.fromstring(
			ctx.encrypt(_make_ctr_block(nonce, numpy.uint64(counter))),
			dtype=numpy.uint8
		)
		for counter in range(num_blocks)
	])
	return s ^ keystream[:s.size]

ctr_decrypt = ctr_encrypt

