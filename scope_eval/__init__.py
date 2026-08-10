"""Surgical camera evaluation research pipeline."""

import os


# Some hosted images export ``OMP_NUM_THREADS=0``.  libgomp rejects that value
# before the CLI has a chance to initialise the numerical stack.
if not os.environ.get("OMP_NUM_THREADS", "").isdigit() or int(os.environ.get("OMP_NUM_THREADS", "0")) < 1:
    os.environ["OMP_NUM_THREADS"] = "1"

__version__ = "0.1.0"
