"""
    compress.py
    ~~~~~~~~~~~
    Author: Tyler Lin

    This script uses a trained model to compress and decompress given PGN data.
"""

from compr_ch_ess import Compressor
from os import system, name
import chess.pgn
import io
import sys


def clear():
    """Clears the terminal screen
    """
    if name == "nt":
        _ = system("cls")
    else:
        _ = system("clear")

    print("\n\n" + "~" * 50)
    print("\t\t compress.py")
    print("~" * 50 + "\n\n")


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: python3 scripts/compress.py <engine_path> <model_path>")
        exit(1)

    engine_path = sys.argv[1]
    model_path = sys.argv[2]

    compressor = Compressor(engine_path)
    compressor.load_model_file(model_path)

    while True:

        clear()

        print("Options:")
        print("1. Compress")
        print("2. Decompress")
        print("3. Compress and Decompress")
        print("4. Exit")

        choice = input("\nChoice: ")

        if choice == "1":
            clear()
            print("Copy and paste the PGN to compress below (ctl-D to finish):")
            pgn = sys.stdin.read()
            game = chess.pgn.read_game(io.StringIO(pgn))
            compressed = compressor.compress(game)
            print("\n\nCompressed in " + str(len(compressed)) + " bytes")
            print("\n\n" + compressed)
        elif choice == "2":
            clear()
            print(
                "Copy and paste the compressed code to decompress below (ctl-D to finish):")
            pgn = sys.stdin.read()
            pgn.replace("\n", "")
            game = compressor.decompress(pgn)
            print(game, end="\n\n")

        elif choice == "3":
            clear()
            print("Copy and paste the PGN to compress below (ctl-D to finish):")
            pgn = sys.stdin.read()
            game = chess.pgn.read_game(io.StringIO(pgn))
            compressed = compressor.compress(game)
            print("\n\nCompressed in " + str(len(compressed)) + " bytes")
            print("\n\n" + compressed + "\n\n")
            game = compressor.decompress(compressed)
            print("\n\nDecompressed:\n\n")
            print(game, end="\n\n")

        elif choice == "4":
            exit(0)

        else:
            print("Invalid choice")

        input("\nPress enter to continue...")
