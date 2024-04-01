import os
from distutils.util import strtobool


class EnvVar(object):
    APP_VERSION = "0.0.1-alpha"
    APP_NAME = "omllava"

    API_PREFIX = os.environ.get('API_PREFIX', '/omllava')
    IS_DEBUG = bool(os.environ.get('IS_DEBUG', False))
    
    NUMGPUS = int(os.environ.get('NUMGPUS', 1))
    GPU_MEMORY_UTILIZATION = float(os.environ.get('GPU_MEMORY_UTILIZATION', 0.7))
    MAX_NUM_BATCHED_TOKENS = int(os.environ.get('MAX_NUM_BATCHED_TOKENS',10000))

