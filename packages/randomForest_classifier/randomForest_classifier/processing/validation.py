from randomForest_classifier.config import config

import pandas as pd
from itertools import compress


def validate_inputs(input_data: pd.DataFrame) -> pd.DataFrame:
    """Check model inputs for unprocessable values."""

    validated_data = input_data.copy()

    # check for numerical variables with NA not seen during training
    if input_data[config.NUMERICAL_NA_NOT_ALLOWED].isnull().any().any():
        validated_data = validated_data.dropna(
            axis=0, subset=config.NUMERICAL_NA_NOT_ALLOWED
        )

    # check for categorical variables with NA not seen during training
    if input_data[config.CATEGORICAL_NA_NOT_ALLOWED].isnull().any().any():
        validated_data = validated_data.dropna(
            axis=0, subset=config.CATEGORICAL_NA_NOT_ALLOWED
        )

    # check for values <= 0 for the log transformed variables
    if (input_data[config.NUMERICALS_LOG_VARS] <= 0).any().any():
        vars_with_neg_values = list(compress(
                                    config.NUMERICALS_LOG_VARS,
                                    (input_data[config.NUMERICALS_LOG_VARS] <= 0).any().tolist())
        )
        not_allowed = (validated_data[vars_with_neg_values] < 0).any(axis = 1)
        validated_data = validated_data[~not_allowed]

    return validated_data
