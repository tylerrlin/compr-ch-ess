"""
    build_model.py
    ~~~~~~~~~~~
    Author: Tyler Lin

    This script helps build a model that can be used to compress and decompress
    chess games. It uses a chess engine to analyze the position and uses the
    analysis to build a dataframe. The dataframe is then used to build a model
    using Keras.
"""

from compr_ch_ess import Model, Data
import chess.pgn

NUM_GAMES = 100

TIME_CONTROL = "600+0"

ENGINE_PATH = "/opt/homebrew/bin/stockfish"

PGN_PATH = "scripts/pgns/games.pgn"

MIN_RATING = 2000
MAX_RATING = 2200

if __name__ == "__main__":

    output_df_path = f"scripts/data/comprchess_{MIN_RATING}_{MAX_RATING}_df.csv"
    output_model_path = f"scripts/models/comprchess_{MIN_RATING}_{MAX_RATING}_pkl"

    print("\n\n" + "~" * 50)
    print("\t\t build_model.py")
    print("~" * 50 + "\n\n")

    games = []
    pgn_data = open(PGN_PATH)
    while len(games) < NUM_GAMES:
        game = chess.pgn.read_game(pgn_data)
        if ((int(game.headers["WhiteElo"]) + int(game.headers["BlackElo"])) / 2 >= MIN_RATING and (int(game.headers["WhiteElo"]) + int(game.headers["BlackElo"])) / 2 <= MAX_RATING and game.headers["TimeControl"] == TIME_CONTROL):
            games.append(game)

    print(f"Loaded {len(games)} games")
    print("\nBuilding dataframe...\n")

    data = Data(ENGINE_PATH)
    dataframe = data.build_dataframe_from_games(games)

    print("\nSaving dataframe...\n")
    data.save_dataframe(dataframe, output_df_path)

    print("\nTraining model...\n")

    num_train = int(len(dataframe) * 0.8)
    if num_train % 2 == 1:
        num_train += 1

    train_data = dataframe.iloc[:int((num_train / 2))]
    val_data = dataframe.iloc[int((num_train / 2)):num_train]
    test_data = dataframe.iloc[num_train:]

    model = Model()
    model.build(train_data, val_data)

    print("\nEvaluating model...\n")

    print(model.evaluate(test_data))

    print("\nSaving model...\n")
    model.save(output_model_path)
