NonceMistake
============

A game where you interactively break AES-CTR encrypted fortunes

The mistake made by this program is that the key, nonce and counter are
the same for all fortunes.

Navigate with the arrow keys, and enter the character you want in the
highlighted cell.  The rest of the row will update to suit.

The ctrl key will toggle hex mode - all cells will display in hex, and
you may simply enter 2 hex characters to change the value of the current
cell.

We lose the AES key as soon as the ciphertexts are created, so the only
way the game can cheat for you is to do statistical analysis of a column.
Hit F1 to cheat.

If you don't want to cheat, there's still a better way to do this than
guess random characters. Since this is an XOR cipher, you can consider
the bits to be columns - for letters bit 5 is uppercase or lowercase, and
for all ASCII, bit 7 is clear.  Try hex mode, it may help if you're that
way inclined. CTRL toggles hex mode.

REQUIREMENTS:

- Python 2.x, where x is probably 7
- Pygame
- python-crypto
- "fortune" installed and in your path.  Get more than just fortunes-min
  if your flight is expected to take more than an hour.
- A system font calls "Sans"

Enjoy, share, have fun, and learn from others' mistakes.
