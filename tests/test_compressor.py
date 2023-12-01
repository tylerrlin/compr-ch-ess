from compr_ch_ess import Model


def test_chess():
    cm = Model("/opt/homebrew/bin/stockfish")

    # convert pgn file to string
    pgn = open("tests/test.pgn")
    pgn_string = pgn.read()

    cm.train_pgns.append(pgn_string)
    cm.build_dataframe()
    print(cm.dataframe)

    assert False
