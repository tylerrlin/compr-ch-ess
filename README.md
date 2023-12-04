# COMPR\[CH]ESS

## Introduction

COMPR\[CH]ESS is an innovative approach to lossless chess game compression, seamlessly blending chess engines, machine learning, and Huffman Coding. By training models that predict player moves at a given rating range, this program can adaptively
encode/decode chess games efficiently.

## Getting Started

Follow these simple steps to get started with COMPR[CH]ESS:

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/tylerrlin/compr_ch_ess.git
    cd compr_ch_ess
    ```
2. **Install Package:**
    ```bash
    python setup.py install
    ```
3. **Run a Script:**

    ```bash
    python scripts/compress.py <engine_path> <model_path>
    ```

    #### or to build a model (will have to edit the default values in scripts/build_model.py):

    ```bash
    python scripts/build_model.py <engine_path>
    ```

## How it works

This program utilizes Huffman Coding to encode/decode chess games. Huffman Coding is a technique where more probable characters
or strings of characters are assigned lower-length and unique bitstrings. Using a chess engine to evaluate positions and a machine learning model trained to predict a player's next move, every unique move is encoded and appended to a bitstring containing the code of the whole game. For a hard-coded visualization of this, check out [https://tylerrlin.github.io/projects/comprchess](https://tylerrlin.github.io/projects/comprchess).
