import numpy as np
import pandas as pd

from randomForest_classifier.processing.data_management import load_pipeline
from randomForest_classifier.config import config
from randomForest_classifier.processing.validation import validate_inputs
from randomForest_classifier import __version__ as _version

import logging
import typing as t


_logger = logging.getLogger(__name__)

pipeline_file_name = f"{config.PIPELINE_SAVE_FILE}{_version}.pkl"
_mora_pipe = load_pipeline(file_name=pipeline_file_name)


def make_prediction(*, input_data: t.Union[pd.DataFrame, dict],
                    ) -> dict:
    """Make a prediction using a saved model pipeline.

    Args:
        input_data: Array of model prediction inputs.

    Returns:
        Predictions for each input row, as well as the model version.
    """

    data = pd.DataFrame(input_data)
    validated_data = validate_inputs(input_data=data)

    #print('Parameters')
    #print(_mora_pipe.get_params())
    prediction = _mora_pipe.predict(validated_data[config.FEATURES])

    #output = np.exp(prediction)
    output = prediction

    results = {"predictions": output, "version": _version}

    _logger.info(
        f"Making predictions with model version: {_version} "
        #f"Inputs: {validated_data} "
        #f"Predictions: {results}"
    )

    return results

def make_transform(*, input_data: t.Union[pd.DataFrame, dict],
                    ) -> dict:
    """Make a transform using a saved model pipeline.

    Args:
        input_data: Array of model prediction inputs.

    Returns:
        Transform for each input row, as well as the model version.
    """

    data = pd.DataFrame(input_data)
    validated_data = validate_inputs(input_data=data)

    #print('Parameters')
    #print(_mora_pipe.get_params())
    prediction = _mora_pipe.transform(validated_data[config.FEATURES])
    
    try:
        target = validated_data[config.TARGET]
    except:
        target = None

    #output = np.exp(prediction)
    output = prediction

    results = {"transform": output, "target": target, "version": _version}

    _logger.info(
        f"Making predictions with model version: {_version} "
        #f"Inputs: {validated_data} "
        #f"Predictions: {results}"
    )

    return results

