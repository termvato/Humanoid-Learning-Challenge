"""Submit a training job to the deployed `handsim-train` app. Spawned calls run on Modal independently of this
laptop's connection (a `modal run --detach` was cancelled when the local client lost its connection).

Usage: .venv/bin/modal deploy infra/modal_train.py   # after any change to the app
       .venv/bin/python infra/modal_launch.py <dataset> <run> "<lerobot-train args>"
Prints the call id; the run writes /vol/train/<run> and /vol/train/<run>.log in the `handsim` volume.
"""

import sys

import modal

dataset, run, args = sys.argv[1:4]
call = modal.Function.from_name("handsim-train", "train").spawn(dataset, run, args)
print(run, call.object_id)
