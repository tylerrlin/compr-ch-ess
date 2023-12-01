"""
    build_model.py
    ~~~~~~~~~~~
    Author: Tyler Lin

    This script helps build a model that can be used to compress and decompress
    chess games. It uses a chess engine to analyze the position and uses the
    analysis to build a dataframe. The dataframe is then used to build a model
    using Keras.
"""

from compr_ch_ess import Compressor, Model, Data
import chess
import chess.pgn

import numpy as np
import pandas as pd
import keras

import pandas as pd

import sys

if __name__ == "__main__":
    # # Configure the data
    # data = pd.read_csv("tests/data.csv")
    # data.drop("Unnamed: 0", axis=1, inplace=True)
    # train = data.iloc[0:20]
    # val = data.iloc[20:40]
    # test = data.iloc[40:60]

    # test = test.drop("result", axis=1)

    # # Configure the model
    # model = Model()
    # model.build(train, val)
    # model.save("tests/model.h5")

    # m = model.get_model()
    # print(test.shape)
    # print(np.array([[1, 2, 3, 4]]).shape)
    # print(m.predict(test))

    # --------------------------

    compressor = Compressor("/opt/homebrew/bin/stockfish")
    compressor.load_model("tests/model.h5")

    pgn = open("tests/test.pgn")
    game = chess.pgn.read_game(pgn)

    code = compressor.compress(game)
    print(len(code))
    print(code)

    game = compressor.decompress(code)
    for move in game.mainline_moves():
        print(move.uci())
