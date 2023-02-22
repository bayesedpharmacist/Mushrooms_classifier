# Suppresses the Tensorflow messages, when executing the shroom file

import os
os.environ['SETUPTOOLS_USE_DISTUTILS'] = 'stdlib'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
