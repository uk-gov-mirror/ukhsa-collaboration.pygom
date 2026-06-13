''' pygom

.. moduleauthor:: Edwin Tye <Edwin.Tye@phe.gov.uk>

'''
import importlib.metadata

from .loss import *
from .model import *
#from .utilR import *

try:
    __version__ = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError:
    # package is not installed
    pass
