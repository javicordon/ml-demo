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
 'numero_cuotas',
 'cl_unq_act_act_messolicitud',
 'cl_unq_act_act_trimestresolicitud',
 'cl_unq_act_act_fechasolicitud',
 'cl_unq_act_act_fechasolicitud_date',
 'cl_unq_act_act_monto',
 'cl_unq_act_act_plazo',
 'cl_unq_act_act_agencia',
 'cl_unq_act_act_ptodestino',
 'cl_unq_act_act_flagaprobado',
 'cl_unq_act_act_longitud',
 'cl_unq_act_act_latitud',
 'cl_unq_act_act_depnacimiento',
 'cl_unq_act_act_estadocivil',
 'cl_unq_act_act_estadocivilmodificado',
 'cl_unq_act_act_genero',
 'cl_unq_act_act_profesion',
 'cl_unq_act_act_profesionmodificada',
 'cl_unq_act_act_flagpuedeescribir',
 'cl_unq_act_act_flagpuedeleer',
 'cl_unq_act_act_flaghablaespa_ol',
 'cl_unq_act_act_flagpuedefirmar',
 'cl_unq_act_act_flaghablaotroidioma',
 'cl_unq_act_act_nivelacademico',
 'cl_unq_act_act_tiempovivirresidencia',
 'cl_unq_act_act_tipovivienda',
 'cl_unq_act_act_personasdependientes',
 'cl_unq_act_act_tipolocalidad',
 'cl_unq_act_act_topografia',
 'cl_unq_act_act_flagaccesovehicular',
 'cl_unq_act_act_tipoaccesovehicular',
 'cl_unq_act_act_tipoaccesopeatonal',
 'cl_unq_act_act_flagaccesomensajeros',
 'cl_unq_act_act_flagpidenimpuesto',
 'cl_unq_act_act_vivtipoconstruccion',
 'cl_unq_act_act_cantidadniveles',
 'cl_unq_act_act_cantidaddormitorios',
 'cl_unq_act_act_cantidadba_os',
 'cl_unq_act_act_flagtienecocina',
 'cl_unq_act_act_flagtienesala',
 'cl_unq_act_act_flagtienejardin',
 'cl_unq_act_act_flagtienegarage',
 'cl_unq_act_act_flagtienecomedor',
 'cl_unq_act_act_vehiculo',
 'cl_unq_act_act_fuenteingresos',
 'cl_unq_act_act_tiponegocio',
 'cl_unq_act_act_depnegocio',
 'cl_unq_act_act_flagvendealcredito',
 'cl_unq_act_act_negociomontoventasefectivo',
 'cl_unq_act_act_negociototalingresos',
 'cl_unq_act_act_totalbienes',
 'cl_unq_act_act_totalpasivos',
 'cl_unq_act_act_totalgastosfam',
 'cl_unq_act_act_totalingresosfam',
 'cl_unq_act_act_estresventas',
 'cl_unq_act_act_estrescostoventas',
 'cl_unq_act_act_estresgrossprofit',
 'cl_unq_act_act_flagtieneelectricidad',
 'cl_unq_act_act_flagtieneagua',
 'cl_unq_act_act_flagtienetelfijo',
 'cl_unq_act_act_flagtienecelular',
 'cl_unq_act_act_flagtienetvcable',
 'cl_unq_act_act_flagtienerefrigerador',
 'cl_unq_act_act_flagtienelavadora',
 'cl_unq_act_act_flagtienesecadora',
 'cl_unq_act_act_flagtienehorno',
 'cl_unq_act_act_flagtienemicroondas',
 'cl_unq_act_act_flagtienestereo',
 'cl_unq_act_act_fnacimiento_date',
]

# this variable is to calculate the temporal variable,
# can be dropped afterwards
DROP_FEATURES = [
    "cl_unq_act_act_fechasolicitud",
]

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
    "cl_unq_act_act_profesionmodificada",
]

TEMPORAL_VARS = "cl_unq_act_act_fnacimiento_date"

# variables to log transform
NUMERICALS_LOG_VARS = [
    #"cl_unq_act_act_totalgastosfam"
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
    'cl_unq_act_act_fechasolicitud_date',
    'cl_unq_act_act_agencia',
    'cl_unq_act_act_profesion',
    'cl_unq_act_act_flaghablaespa_ol',
    'cl_unq_act_act_flagpuedefirmar',
    'cl_unq_act_act_flaghablaotroidioma',
    'cl_unq_act_act_nivelacademico',
    'cl_unq_act_act_tipovivienda',
    'cl_unq_act_act_tipolocalidad',
    'cl_unq_act_act_topografia',
    'cl_unq_act_act_tipoaccesopeatonal',
    'cl_unq_act_act_flagpidenimpuesto',
    'cl_unq_act_act_vivtipoconstruccion',
    'cl_unq_act_act_flagtienecocina',
    'cl_unq_act_act_flagtienesala',
    'cl_unq_act_act_flagtienejardin',
    'cl_unq_act_act_vehiculo',
    'cl_unq_act_act_fuenteingresos',
    'cl_unq_act_act_tiponegocio',
    'cl_unq_act_act_flagvendealcredito',
    'cl_unq_act_act_flagtieneelectricidad',
    'cl_unq_act_act_flagtienetelfijo',
    'cl_unq_act_act_flagtienecelular',
    'cl_unq_act_act_flagtienetvcable',
    'cl_unq_act_act_flagtienesecadora',
    'cl_unq_act_act_flagtienehorno',
    'cl_unq_act_act_flagtienemicroondas',
    "cl_unq_act_act_profesionmodificada",
]

NUMERICAL_NA_NOT_ALLOWED = [
    #feature
    #for feature in FEATURES
    #if feature not in CATEGORICAL_VARS + NUMERICAL_VARS_WITH_NA
]

CATEGORICAL_NA_NOT_ALLOWED = [
    #feature for feature in CATEGORICAL_VARS if feature not in CATEGORICAL_VARS_WITH_NA
]

DICTIONARY_REPLACER = {
    "cl_unq_act_act_totalgastosfam": {"target": 0, "replace_value": 1},
    #"cl_unq_act_act_negociototalingresos": {"target": 0, "replace_value": 2},
    "cl_unq_act_act_profesion": {"target":'domesticos', "replace_value":'oficios_domesticos',
                                 "target":'domestica', "replace_value": 'oficios_domesticos',
                                 "target":'domestico', "replace_value": 'oficios_domesticos',
                                 "target":'oficio_domesticos', "replace_value": 'oficios_domesticos',
                                 "target":'Comercializador', "replace_value": 'comerciante',
                                 "target":'mer', "replace_value": 'comerciante',
                                 "target":'iante', "replace_value": 'comerciante',
                                 "target":'merciante', "replace_value": 'comerciante',
                                 "target":'maestros', "replace_value": 'maestro',
                                 "target":'profesor', "replace_value": 'maestro',
                                 "target":'maestra', "replace_value": 'maestro',
                                 "target":'maesto(a)', "replace_value": 'maestro',
                                 "target":'prof', "replace_value": 'maestro',
                                 "target":'casa', "replace_value": 'ama_de_casa',
                                 "target":'ama', "replace_value": 'ama_de_casa',
                                 "target":'hogar', "replace_value": 'ama_de_casa',
                                 "target":'madre', "replace_value": 'ama_de_casa',
                                 "target":'agricultura', "replace_value": 'agricultor',
                                 "target":'agro', "replace_value": 'agricultor',
                                 "target":'estudia', "replace_value": 'estudiante',
                                 "target":'estudio', "replace_value": 'estudiante',
                                 "target":'universitario', "replace_value": 'estudiante'},
    "cl_unq_act_act_estadocivil": {"target":'union', "replace_value":'unionlibre',
                                   "target":'union_libre', "replace_value":'unionlibre',
                                   "target":'unido', "replace_value":'unionlibre',
                                   "target":'libre', "replace_value":'unionlibre',
                                   "target":'unido_d', "replace_value":'unionlibre',
                                   "target":'unida_d', "replace_value":'unionlibre',
                                   "target":'soltera', "replace_value":'soltero',
                                   "target":'soltero(a)', "replace_value":'soltero',
                                   "target":'casada', "replace_value":'casado',
                                   "target":'casado(a)', "replace_value":'soltero',
                                   "target":'divorciada', "replace_value":'divorciado',
                                   "target":'divorciado(a)', "replace_value":'divorciado',
                                   "target":'div', "replace_value":'divorciado'}
}

PIPELINE_NAME = "randomForest_classifier"
PIPELINE_SAVE_FILE = f"{PIPELINE_NAME}"
BEST_PARAMS_SAVE_FILE = f"{PIPELINE_NAME}_bestParams"

# used for differential testing
ACCEPTABLE_MODEL_DIFFERENCE = 0.05

# used to save variables who can be modified
MODIFIED_VARS = {"cl_unq_act_act_estadocivil": {"feature_modified": "cl_unq_act_act_estadocivilmodificado"},
                 "cl_unq_act_act_profesion": {"feature_modified": "cl_unq_act_act_profesionmodificada"}}