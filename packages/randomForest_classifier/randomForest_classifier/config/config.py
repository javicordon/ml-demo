import pathlib

import randomForest_classifier

import pandas as pd


pd.options.display.max_rows = 10
pd.options.display.max_columns = 10


PACKAGE_ROOT = pathlib.Path(randomForest_classifier.__file__).resolve().parent
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"
DATASET_DIR = PACKAGE_ROOT / "datasets"

# data
TESTING_DATA_FILE = "test.csv"
TRAINING_DATA_FILE = "train.csv"
DATA_FILE = "dataset.csv"
TARGET = "maxmora"


# variables
FEATURES = [
    "cl_unq_act_act_messolicitud",
    "cl_unq_act_act_ptodestino",
    "cl_unq_act_act_depnacimiento",
    "cl_unq_act_act_estadocivil",
    "cl_unq_act_act_estadocivilmodificado",
    "cl_unq_act_act_genero",
    "cl_unq_act_act_flagpuedeescribir",
    "cl_unq_act_act_flagpuedeleer",
    "cl_unq_act_act_tiempovivirresidencia",
    "cl_unq_act_act_flagaccesovehicular",
    "cl_unq_act_act_tipoaccesovehicular",
    "cl_unq_act_act_flagaccesomensajeros",
    "cl_unq_act_act_flagtienegarage",
    "cl_unq_act_act_flagtienecomedor",
    "cl_unq_act_act_depnegocio",
    "cl_unq_act_act_flagtieneagua",
    "cl_unq_act_act_flagtienerefrigerador",
    "cl_unq_act_act_flagtienelavadora",
    "cl_unq_act_act_flagtienestereo",
    "cl_unq_act_act_totalgastosfam",
    "cl_unq_act_act_negociototalingresos",
    "cl_unq_act_act_totalbienes",
    "cl_unq_act_act_monto",
    # this one is only to calculate temporal variable:
    "cl_unq_act_act_fechasolicitud",
    "cl_unq_act_act_fnacimiento_date",
]

# this variable is to calculate the temporal variable,
# can be dropped afterwards
DROP_FEATURES = "cl_unq_act_act_fechasolicitud"

# numerical variables with NA in train set
NUMERICAL_VARS_WITH_NA = [
    "cl_unq_act_act_totalgastosfam",
    "cl_unq_act_act_negociototalingresos",
    "cl_unq_act_act_totalbienes",
    "cl_unq_act_act_monto"
]

# categorical variables with NA in train set
CATEGORICAL_VARS_WITH_NA = [
    "cl_unq_act_act_depnacimiento",
    "cl_unq_act_act_estadocivil",
    "cl_unq_act_act_estadocivilmodificado",
    "cl_unq_act_act_genero",
    "cl_unq_act_act_tiempovivirresidencia",
    "cl_unq_act_act_tipoaccesovehicular",
    "cl_unq_act_act_depnegocio",
]

TEMPORAL_VARS = "cl_unq_act_act_fnacimiento_date"

# variables to log transform
NUMERICALS_LOG_VARS = [
    "cl_unq_act_act_totalgastosfam",
    "cl_unq_act_act_negociototalingresos",
    "cl_unq_act_act_totalbienes",
    "cl_unq_act_act_monto"
]

# categorical variables to encode
CATEGORICAL_VARS = [
    "cl_unq_act_act_ptodestino",
    "cl_unq_act_act_depnacimiento",
    "cl_unq_act_act_estadocivil",
    "cl_unq_act_act_estadocivilmodificado",
    "cl_unq_act_act_genero",
    "cl_unq_act_act_flagpuedeescribir",
    "cl_unq_act_act_flagpuedeleer",
    "cl_unq_act_act_tiempovivirresidencia",
    "cl_unq_act_act_flagaccesovehicular",
    "cl_unq_act_act_tipoaccesovehicular",
    "cl_unq_act_act_flagaccesomensajeros",
    "cl_unq_act_act_flagtienegarage",
    "cl_unq_act_act_flagtienecomedor",
    "cl_unq_act_act_depnegocio",
    "cl_unq_act_act_flagtieneagua",
    "cl_unq_act_act_flagtienerefrigerador",
    "cl_unq_act_act_flagtienelavadora",
    "cl_unq_act_act_flagtienestereo",
]

NUMERICAL_NA_NOT_ALLOWED = [
    feature
    for feature in FEATURES
    if feature not in CATEGORICAL_VARS + NUMERICAL_VARS_WITH_NA
]

CATEGORICAL_NA_NOT_ALLOWED = [
    feature for feature in CATEGORICAL_VARS if feature not in CATEGORICAL_VARS_WITH_NA
]

DICTIONARY_REPLACER = {
    "cl_unq_act_act_totalgastosfam": {"target": 0, "replace_value": 1},
    #"cl_unq_act_act_negociototalingresos": {"target": 0, "replace_value": 2},
}


PIPELINE_NAME = "randomForest_classifier"
PIPELINE_SAVE_FILE = f"{PIPELINE_NAME}_output_v"

# used for differential testing
ACCEPTABLE_MODEL_DIFFERENCE = 0.05
