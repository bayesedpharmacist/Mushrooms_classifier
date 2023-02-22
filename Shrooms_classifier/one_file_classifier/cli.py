# Runs the predict.py

import os
from Shrooms_classifier.__main__ import main

os.environ['SETUPTOOLS_USE_DISTUTILS'] = 'stdlib'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# print(dir(main))

if __name__ == "__main__":
    main()

