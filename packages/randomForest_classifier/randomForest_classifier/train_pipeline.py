import numpy as np
from sklearn.model_selection import train_test_split

from randomForest_classifier import pipeline
from randomForest_classifier.processing.data_management import load_dataset, save_dataset, save_pipeline
from randomForest_classifier.config import config
from randomForest_classifier import __version__ as _version

import logging

_logger = logging.getLogger(__name__)


def run_training() -> None:
    """Train the model."""

    # read training data
    data = load_dataset(file_name=config.DATA_FILE)

    # replace variables with dictionary
    for feature in config.DICTIONARY_REPLACER.keys():
        data[feature] = data[feature].replace(
                                    config.DICTIONARY_REPLACER[feature]["target"],
                                    config.DICTIONARY_REPLACER[feature]["replace_value"])

    # remove zeroes and negatives
    ver = ~(data[config.NUMERICALS_LOG_VARS] <= 0).any(axis = 1)
    data = data.loc[ver]

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.FEATURES], data[config.TARGET], test_size=0.1, random_state=0
    )  # we are setting the seed here

    # save training and testing
    save_dataset(file_name=config.TRAINING_DATA_FILE, df=X_train)
    save_dataset(file_name=config.TESTING_DATA_FILE, df=X_test)

    # transform the target
    #y_train = np.log(y_train)

    pipeline.mora_pipe.fit(X_train[config.FEATURES], y_train)
    #pred = pipeline.mora_pipe.predict(X_train[config.FEATURES])
    #print("OUTPUT PRED SUM",pred.sum())

    _logger.info(f"saving model version: {_version}")
    save_pipeline(pipeline_to_persist=pipeline.mora_pipe)


if __name__ == "__main__":
    run_training()
