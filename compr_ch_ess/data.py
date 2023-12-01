"""
    data.py
    ~~~~~~~~~~~
    Author: Tyler Lin

    This module implements the Data class. This class is used to help build
    dataframes from chess games using a engine analysis. The user provides
    PGN files and the Data class will build a dataframe from the games.
"""

import chess.engine
import chess.pgn

import pandas as pd
import numpy as np

import io

PIECE_VALUES = {"K": 0, "Q": 900, "R": 500, "B": 325, "N": 300, "P": 100,
                "k": 0, "q": -900, "r": -500, "b": -325, "n": -300, "p": -100}


class Data:

    def __init__(self, engine_path: str) -> None:
        """Initializes the Model class

        :param engine_path: The path to the utilized chess engine
        :type engine_path: str

        :return: None
        """
        self.__engine_path = engine_path
        self.__engine_depth = 10
        self.__mate_score = 5000

    def configure_engine(self, depth: int, mate_score: int) -> None:
        """Configures the chess engine

        :param depth: The depth to analyze the game to
        :type depth: int
        :param mate_score: The mate score to use for the engine
        :type mate_score: int

        :return: None
        """
        self.__engine_depth = depth
        self.__mate_score = mate_score

    def _meets_filters(self, game: chess.pgn.Game, filters: dict) -> bool:
        """Determines if the given game meets the given filters

        :param game: The game to check
        :type game: chess.pgn.Game
        :param filters: The filters to check
        :type filters: dict

        :return: True if the game meets the filters, False otherwise
        :rtype: bool
        """
        if filters is None:
            return True
        for key in filters:
            if key not in game.headers():
                raise Exception(f"Invalid filter: {key}")
            if game.headers()[key] != filters[key]:
                return False
        return True

    def analyze_game(self, game: chess.pgn.Game) -> list:
        """Analyzes the given game using specified chess engine

        :param game: The game to analyze
        :type game: chess.pgn.Game

        :return: Data from the analyzed game
        :rtype: list
        """
        game_data = []
        board = game.board()
        # Initialize the engine
        engine = chess.engine.SimpleEngine.popen_uci(self.__engine_path)
        for move_number, played in enumerate(game.mainline_moves()):
            # determine the number of legal moves
            legal_moves = board.legal_moves.count()
            # determine the point difference
            point_diff = 0
            for piece in board.piece_map():
                point_diff += PIECE_VALUES[board.piece_map()[piece].symbol()]
            # analyze the position
            analysis = engine.analyse(board, chess.engine.Limit(
                depth=self.__engine_depth), multipv=board.legal_moves.count())
            # determine the engine's evaluation of the position
            eval = analysis[0]["score"].relative.score(
                mate_score=self.__mate_score)
            # determine the engine's rank of the move played
            result = next(move["multipv"]
                          for move in analysis if move["pv"][0] == played)
            # add the data to the list
            game_data.append(
                {
                    "result": result,
                    "eval": eval,
                    "move_number": move_number,
                    "legal_moves": legal_moves,
                    "point_diff": point_diff if board.turn else -point_diff,
                }
            )
            # update the board with the move played
            board.push(played)

        engine.quit()
        return game_data

    def build_dataframe_from_games(self, games: list) -> pd.DataFrame:
        """Builds a dataframe from a given list of games

        :param games: The list of games to build the dataframe from
        :type games: list of chess.pgn.Game

        dataframe columns:
            - result: rank of the move played (determined by the engine)
            - eval: evaluation of the position before the move was played
            - move_number: the move number
            - legal_moves: the possible moves from the position
            - point_diff: the difference in points between the two players

            ** both eval and point_diff are relative to the player who played the move

        :return: The dataframe built from the games
        :rtype: pd.DataFrame
        """
        # initialize the data list
        data = []
        for i, game in enumerate(games):
            print(f"Analyzing game {i + 1} of {len(games)}")
            data.extend(self.analyze_game(game))
        # check if no games were found
        if len(data) == 0:
            raise Exception("No games to build the dataframe from")
        # print success message and return
        print("\nDataframe built successfully")
        return pd.DataFrame(data)

    def build_dataframe_from_strs(self, pgns: list) -> pd.DataFrame:
        """Builds a dataframe from a given list of PGNs

        :param pgns: The list of PGNs to build the dataframe from
        :type pgns: list

        dataframe columns:
            - result: rank of the move played (determined by the engine)
            - eval: evaluation of the position before the move was played
            - move_number: the move number
            - legal_moves: the possible moves from the position
            - point_diff: the difference in points between the two players

            ** both eval and point_diff are relative to the player who played the move

        :return: The dataframe built from the PGNs
        :rtype: pd.DataFrame
        """
        # initialize the data list
        data = []
        for i, pgn in enumerate(pgns):
            game = chess.pgn.read_game(io.StringIO(pgn))
            if game is None:
                raise Exception(f"Invalid PGN given (index: {i}):\n\n{pgn}")
            print(f"Analyzing game {i + 1} of {len(pgns)}")
            data.extend(self.analyze_game(game))
        # check if no games were found
        if len(data) == 0:
            raise Exception("No games to build the dataframe from")
        # print success message and return
        print("\nDataframe built successfully")
        return pd.DataFrame(data)

    def build_dataframe_from_path(self, pngs_path: str, num_games: int, filters: dict = None) -> pd.DataFrame:
        """Builds a dataframe from a given path to a PGN file

        :param pngs_path: The path to the PGN file
        :type pngs_path: str
        :param filters: The filters to apply to the PGN file (PGN headers)
                        ex. {"TimeControl": "180+2", "White": "username"}
        :type filters: dict

        dataframe columns:
            - result: rank of the move played (determined by the engine)
            - eval: evaluation of the position before the move was played
            - move_number: the move number
            - legal_moves: the possible moves from the position
            - point_diff: the difference in points between the two players

            ** both eval and point_diff are relative to the player who played the move

        :return: The dataframe built from the PGNs
        :rtype: pd.DataFrame
        """
        # initialize the data list
        data = []
        with open(pngs_path) as pgn_file:
            game = chess.pgn.read_game(pgn_file)
            while game is not None and len(data) < num_games:
                # check if the game meets other filters
                if self._meets_filters(game, filters):
                    print(f"Analyzing game {len(data) + 1} of {num_games}")
                    data.extend(self.analyze_game(game))
                game = chess.pgn.read_game(pgn_file)
            # print number of games found
            if len(data) < num_games:
                print(f"Found {len(data)} games of {num_games} requested")
        # check if no games were found
        if len(data) == 0:
            raise Exception("No games to build the dataframe from")
        # print success message and return
        print("\nDataframe built successfully")
        return pd.DataFrame(data)

    def save_dataframe(self, dataframe: pd.DataFrame, path: str) -> None:
        """Saves the given dataframe to the given path

        :param dataframe: The dataframe to save
        :type dataframe: pd.DataFrame
        :param path: The path to save the dataframe to
        :type path: str

        :return: None
        """
        dataframe.to_csv(path, index=False)
        print(f"Dataframe saved succesfully to {path}")
