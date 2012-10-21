# Copyright (C) 2012 Brian Wesley Baugh
import os
import sys
import inspect


# Add the project's lib/ folder to the python path. This makes it so
# that if the required libraries are available in the lib/ directory
# they will be used even if they were not installed into Python's
# site-packages/ directory.
# Note: the lib_dir var is still available if needed from the outside.
lib_dir = os.path.realpath(os.path.abspath(os.path.join(os.path.split(
    inspect.getfile(inspect.currentframe()))[0], "../lib")))
if lib_dir not in sys.path:
    sys.path.insert(0, lib_dir)
