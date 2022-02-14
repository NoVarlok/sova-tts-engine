import sys
import os

import_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, import_path)

import modules
import utils
import data

from model import load_model as load_tacotron_model
from . import hparams


sys.path.pop(0)
