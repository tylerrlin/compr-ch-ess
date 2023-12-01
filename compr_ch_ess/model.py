"""
    model.py
    ~~~~~~~~~~~
    Author: Tyler Lin

    This module implements the Model class. This class is used to build a 
    model that measures probability of a move being played given a position.
"""

import keras

import pandas as pd
import numpy as np

import pickle


class Model:

    def __init__(self) -> None:
        """Initializes the Model class

        :return: None
        """
        self.__model = None
        self.__train_epochs = 10
        self.__train_batch_size = 32
        self.__eval_batch_size = 32
        self.__max_legal_moves = 128

    def get_model(self) -> keras.Sequential:
        """Returns the model

        :return: The model
        :rtype: keras.Sequential
        """
        return self.__model

    def configure(self, train_epochs: int, train_batch_size: int, eval_batch_size: int, max_legal_moves: int) -> None:
        """Configures the model

        :param train_epochs: The number of epochs to train the model for
        :type train_epochs: int
        :param train_batch_size: The batch size to train the model with
        :type train_batch_size: int
        :param eval_batch_size: The batch size to evaluate the model with
        :type eval_batch_size: int
        :param max_legal_moves: The maximum number of legal moves in a position
        :type max_legal_moves: int

        :return: None
        """
        self.__train_epochs = train_epochs
        self.__train_batch_size = train_batch_size
        self.__eval_batch_size = eval_batch_size
        self.__max_legal_moves = max_legal_moves

    def build(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> None:
        """Builds the model from the given dataframe using Keras

        :param train_data: The training data to build the model from
        :type train_data: pd.DataFrame
        :param val_data: The validation data to build the model from
        :type val_data: pd.DataFrame

        :return: None
        """
        # Set a random seed for reproducibility ("CHESS" in ASCII !)
        np.random.seed(1231)
        # Split the data into features and results
        train_features = train_data.drop(["result"], axis=1)
        train_results = train_data["result"]
        validate_features = val_data.drop(["result"], axis=1)
        validate_results = val_data["result"]
        # Build the model
        self.__model = keras.Sequential([
            keras.layers.Dense(64, activation="relu", input_shape=(4,)),
            keras.layers.Dense(128),
            keras.layers.Dense(128),
            keras.layers.Dense(64),
            keras.layers.Dense(64),
            keras.layers.Dense(128),
            keras.layers.Dense(self.__max_legal_moves, activation="softmax")
        ])
        # Compile the model
        self.__model.compile(
            optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        # Fit the model
        self.__model.fit(train_features, train_results, epochs=self.__train_epochs,
                         batch_size=self.__train_batch_size, validation_data=(validate_features, validate_results))
        print("Model built successfully")

    def evaluate(self, test_data: pd.DataFrame) -> list:
        """Evaluates the model on the given test data

        :param test_data: The test data to evaluate the model on
        :type test_data: pd.DataFrame

        :return: The evaluation metrics
        :rtype: list
        """
        # check if the model has been built
        if self.__model is None:
            raise Exception("Model has not been built yet")
        # Split the data into features and results
        test_features = test_data.drop(["result"], axis=1)
        test_results = test_data["result"]
        # Evaluate the model and return the metrics
        return self.__model.evaluate(test_features, test_results, batch_size=self.__eval_batch_size)

    def predict(self, position: np.ndarray) -> np.ndarray:
        """Predicts the probability of each move being played given the given position

        :param position: The position to predict the move probabilities for
        :type position: np.ndarray

        :return: The predicted move probabilities
        :rtype: np.ndarray
        """
        # check if the model has been built
        if self.__model is None:
            raise Exception("Model has not been built yet")
        # Predict the move probabilities and return
        return self.__model.predict(position)

    def save(self, path: str) -> None:
        """Saves the model to the given path

        :param path: The path to save the model to
        :type path: str

        :return: None
        """
        # check if the model has been built
        if self.__model is None:
            raise Exception("Model has not been built yet")
        # Save the model and print success message
        with open(path, 'wb') as file_path:
            pickle.dump(self.__model, file_path)

        print(f"Model saved successfully to {path}")
