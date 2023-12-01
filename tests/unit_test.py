"""
    unit_test.py
    ~~~~~~~~~~~
    Author: Tyler Lin

    This script uses pytest to test the functionality of the compr_ch_ess 
    package. 
"""

from compr_ch_ess import Data, Model, Compressor

import chess
import chess.engine
import chess.pgn

import pandas as pd
import numpy as np


def test_data():
    data = Data()
