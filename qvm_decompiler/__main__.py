from sys import argv

from .ast import *
from .cfg import build_cfg
from .qvm import Qvm
from .decompile import decompile

qvm = Qvm(argv[1])
cfg = build_cfg(qvm)
print(decompile(cfg))
