'''Put the repo root on sys.path so scripts/ can be run directly.

Lets `python scripts/train_diffusion.py` work from anywhere, including a Colab
cell, without needing PYTHONPATH or an editable install.
'''
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
