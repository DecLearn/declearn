# coding: utf-8

"""Communication flags definition."""

# vocabulary of different messages that indicate differents step of
# the training algorithm

# bytes
FLAG_WELCOME = 'welcome'
FLAG_INITATE_TRAINING = 'iniating_training'
FLAG_STOP_TRAINING = 'stop_training'
FLAG_REFUSE_CONNECTION = 'connection refused'
FLAG_ERROR = b"error_server"
FLAG_JSON_ERROR = b'key_error'  # error when reading json messages (key missing or mistyped)
FLAG_FATAL_ERROR = b'fatal_error_server'  # indicate that server will not respond to request anymore due to technical failure
FLAG_REQUEST_MODEL_INFO = b"model_info"  # request data info to client(shape, type)

FLAG_INCONSISTANT_MODEL_ERROR = b'model type mismatched'
FLAG_INCONSISTANT_FEATURE_ERROR = b'data shape mismatched'
# status (JSON keys)

FIRST_CONNECTION = "greeting"
TRAINING = "training"
DATA_INFO = "data_info"
DATA_SHAPE = 'data_shape'
