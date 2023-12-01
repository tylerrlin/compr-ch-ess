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


def test_data():
    """ Tests the Data class by building dataframes from a PGN files and
    comparing the results to expected values.
    """
    data = Data("/opt/homebrew/bin/stockfish")

    # configure the engine and make sure the values are set correctly
    data.configure_engine(10, 5000)
    assert (data._Data__engine_depth == 10)
    assert (data._Data__mate_score == 5000)

    # read in a test PGN file into chess board
    test_pgn = open("tests/files/game.pgn")
    test_game = chess.pgn.read_game(test_pgn)
    assert (test_game.headers["Event"] == "It (cat.17)")

    # read in the test PGN file into a string
    test_pgn = open("tests/files/game.pgn")
    test_str = test_pgn.read()

    # build a dataframe from the PGN in two ways and make sure shape is correct
    from_file = data.build_dataframe_from_path("tests/files/game.pgn", 1)
    from_str = data.build_dataframe_from_strs([test_str])
    assert (from_file.shape == (87, 5))
    assert (from_str.shape == (87, 5))

    # make sure the dataframe is built in the same way
    assert (from_file.equals(from_str))

    # test analyzing a game
    analysis = data.analyze_game(test_game)
    assert (len(analysis) == 87)


def test_model():
    """ Tests the Model class by unit testing individual functions and making
    sure predictions are made correctly.
    """
    data = Data("/opt/homebrew/bin/stockfish")
    model = Model()

    # configure the engine and make sure the values are set correctly
    data.configure_engine(10, 5000)
    assert (data._Data__engine_depth == 10)
    assert (data._Data__mate_score == 5000)

    # read in a test PGN file into chess board
    test_pgn = open("tests/files/game.pgn")
    test_game = chess.pgn.read_game(test_pgn)
    assert (test_game.headers["Event"] == "It (cat.17)")

    # build a dataframe from the PGN
    df = data.build_dataframe_from_path("tests/files/game.pgn", 1)

    # build the model and make sure it can be saved
    model.build(df, df)
    model.save("tests/files/model.h5")


def test_compressor():
    """ Tests the Compressor class by testing the compression and decompression
    of a chess game.
    """
    compressor = Compressor("/opt/homebrew/bin/stockfish")

    # load the model and make sure it can be loaded
    compressor.load_model_file("tests/files/model.h5")

    # read in a test PGN file into chess board
    test_pgn = open("tests/files/game.pgn")
    test_game = chess.pgn.read_game(test_pgn)
    assert (test_game.headers["Event"] == "It (cat.17)")

    # compress and decompress the game and make sure it is the same
    code = compressor.compress(test_game)
    game = compressor.decompress(code)

    assert (game.board().fen() == test_game.board().fen())
