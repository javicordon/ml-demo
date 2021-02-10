from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from randomForest_classifier.processing import preprocessors as pp
from randomForest_classifier.processing import features
from randomForest_classifier.config import config

import logging


_logger = logging.getLogger(__name__)


mora_pipe = Pipeline(
    [
        (
            "dictionary_imputer",
            pp.DictionaryImputer(variables=config.DICTIONARY_REPLACER),
        ),
        (
            "categorical_imputer",
            pp.CategoricalImputer(variables=config.CATEGORICAL_VARS_WITH_NA),
        ),
        (
            "numerical_inputer",
            pp.NumericalImputer(variables=config.NUMERICAL_VARS_WITH_NA),
        ),
        (
            "temporal_variable",
            pp.YearsVariableEstimator(
                variables=config.TEMPORAL_VARS),
        ),
        (
            "rare_label_encoder",
            pp.RareLabelCategoricalEncoder(tol=0.01, variables=config.CATEGORICAL_VARS),
        ),
        (
            "categorical_encoder",
            pp.CategoricalEncoder(variables=config.CATEGORICAL_VARS),
        ),
        (
            "log_transformer",
            features.LogTransformer(variables=config.NUMERICALS_LOG_VARS),
        ),
        (
            "drop_features",
            pp.DropUnecessaryFeatures(variables_to_drop=config.DROP_FEATURES),
        ),
        ("scaler", MinMaxScaler()),
        ("RandomForest_Classifier", RandomForestClassifier(max_depth=2, random_state=0)),
    ]
)
