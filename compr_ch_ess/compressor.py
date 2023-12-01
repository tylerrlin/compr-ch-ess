"""
    compressor.py
    ~~~~~~~~~~~
    Author: Tyler Lin

    This module implements the Compressor class. This class is used to
    adaptively compress and decompress chess games using machine learning
    and Huffman Coding.
"""

from tensorflow.python import keras

import numpy as np
import pandas as pd

import chess.engine
import chess.pgn

import pickle
import heapq

PIECE_VALUES = {"K": 0, "Q": 900, "R": 500, "B": 325, "N": 300, "P": 100,
                "k": 0, "q": -900, "r": -500, "b": -325, "n": -300, "p": -100}


class Compressor:

    def __init__(self, engine_path: str) -> None:
        """Initializes the Compressor class

        :param engine_path: The path to the utilized chess engine
        :type engine_path: str

        :return: None
        """
        self.__engine_path = engine_path
        self.__engine_depth = 10
        self.__mate_score = 5000

        self._model = None

    def configure(self, engine_time: float, mate_score: int) -> None:
        """Configures the compressor

        :param engine_time: The time to analyze the position for
        :type engine_time: float
        :param mate_score: The mate score to use for the engine
        :type mate_score: int

        :return: None
        """
        self.__engine_time = engine_time
        self.__mate_score = mate_score

    def load_model_obj(self, model: keras.Sequential) -> None:
        """Loads the given model

        :param model: The model to load
        :type model: keras.Sequential

        :return: None
        """
        self._model = model

    def load_model_file(self, model_path: str):
        """Loads the model from the given path

        :param model_path: The path to load a pickled model from
        :type model_path: str

        :return: None
        """
        with open(model_path, "rb") as f:
            self._model = pickle.load(f)

    def compress(self, game: chess.pgn.Game) -> str:
        """Compresses a chess game

        :param game: The chess game to compress
        :type game: chess.pgn.Game

        :return: The compressed chess game
        :rtype: str
        """
        if self._model is None:
            raise Exception("No model loaded")
        # Initialize the board
        encoded_game = ""
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
            analysis = engine.analyse(
                board, chess.engine.Limit(depth=self.__engine_depth), multipv=legal_moves)
            eval = analysis[0]["score"].relative.score(
                mate_score=self.__mate_score)
            # format the data and get the codes
            data = np.array([[eval, move_number, legal_moves, point_diff]])
            # build the tree
            tree = HuffmanTree(self._model, legal_moves)
            tree.build_tree(data)
            tree.create_codes()
            # get the code for the played move and add it to the encoded game
            result = next(move["multipv"]
                          for move in analysis if move["pv"][0] == played)
            encoded_game += tree.codes[result - 1]
            # update the board
            board.push(played)

        engine.quit()
        return encoded_game

    def decompress(self, encoded_game: str) -> chess.pgn.Game:
        """Decompresses a chess game

        :param game: The compressed chess game to decompress
        :type game: str

        :return: The decompressed chess game
        :rtype: str
        """
        if self._model is None:
            raise Exception("No model loaded")
        # Initialize the board
        board = chess.Board()
        # Initialize the engine
        engine = chess.engine.SimpleEngine.popen_uci(self.__engine_path)
        move_number = 0
        while encoded_game != "":
            # determine the number of legal moves
            legal_moves = board.legal_moves.count()
            # determine the point difference
            point_diff = 0
            for piece in board.piece_map():
                point_diff += PIECE_VALUES[board.piece_map()[piece].symbol()]
            # analyze the position
            analysis = engine.analyse(
                board, chess.engine.Limit(depth=self.__engine_depth), multipv=legal_moves)
            eval = analysis[0]["score"].relative.score(
                mate_score=self.__mate_score)
            # format the data and get the codes
            data = np.array([[eval, move_number, legal_moves, point_diff]])
            # build the tree
            tree = HuffmanTree(self._model, legal_moves)
            tree.build_tree(data)
            tree.create_codes()
            # decode the move
            result, encoded_game = tree.decode(encoded_game)
            # get the move from the engine
            played = next(move["pv"][0]
                          for move in analysis if move["multipv"] == result + 1)
            # update the board
            board.push(played)
            move_number += 1
        # return the decompressed game
        engine.quit()
        return chess.pgn.Game().from_board(board)


class HuffmanTree:
    """Implementation of a Huffman Tree for adaptive compression"""

    class Node:
        """Implementation of a node for the Huffman Tree"""

        def __init__(self, probablity: int, result: int, left=None, right=None) -> None:
            """Initializes the Node class

            :param probablity: The probability of the node
            :type probablity: int
            :param result: The result of the node
            :type result: int
            :param left: The left child of the node
            :type left: Node
            :param right: The right child of the node
            :type right: Node

            :return: None
            """
            self.probability = probablity
            self.result = result
            self.left = left
            self.right = right

        def __lt__(self, other):
            return self.probability <= other.probability

    def __init__(self, model: keras.Sequential, legal_moves: int) -> None:
        """Initializes the HuffmanTree class

        :return: None
        """
        self._model = model
        self._legal_moves = legal_moves
        self.__root = None
        self.codes = {}

    def build_tree(self, data: np.ndarray) -> None:
        """Builds the Huffman tree from the given data

        :param data: The data to build the Huffman tree from
        :type data: np.ndarray

        :return: None
        """
        # get the model's prediction
        prediction = self._model.predict(data)
        # build the tree
        heap = []
        for i in range(self._legal_moves):
            heapq.heappush(heap, (1 - prediction[0][i], self.Node(
                1 - prediction[0][i], i, None, None)))
        # build the tree
        while len(heap) > 1:
            # get the two smallest nodes
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            # create a new node with the two smallest nodes as children
            new_node = self.Node(
                left[0] + right[0], None, left[1], right[1])
            # push the new node to the heap
            heapq.heappush(heap, (new_node.probability, new_node))
        # set the root
        self.__root = heap[0][1]

    def __encode_traverse(self, node: Node, code: str) -> None:
        """Traverses the Huffman tree and creates the codes

        :param node: The node to traverse from
        :type node: Node
        :param code: The code to traverse from
        :type code: str

        :return: None
        """
        if node.left is None and node.right is None:
            self.codes[node.result] = code
            return
        self.__traverse(node.left, code + "0")
        self.__traverse(node.right, code + "1")

    def create_codes(self) -> None:
        """Traverses the Huffman tree and creates the codes

        :return: None
        """
        self.__encode_traverse(self.__root, "")

    def __decode_traverse(self, node: Node, code: str) -> tuple:
        """Traverses the Huffman tree and decodes the given code

        :param node: The node to traverse from
        :type node: Node
        :param code: The code to traverse from
        :type code: str

        :return: The result and the remaining code
        :rtype: tuple
        """
        if node.left is None and node.right is None:
            return node.result, code
        if code[0] == "0":
            return self.__decode_traverse(node.left, code[1:])
        return self.__decode_traverse(node.right, code[1:])

    def decode(self, code: str) -> tuple:
        """Traverses the Huffman tree and decodes the given code

        :param code: The code to traverse from
        :type code: str

        :return: The result and the remaining code
        :rtype: tuple
        """
        return self.__decode_traverse(self.__root, code)
