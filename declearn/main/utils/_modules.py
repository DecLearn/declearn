"""Utils to handle the management and coexistance of Declearn modules"""


class IncompatibleModulesError(Exception):
    """Custom exception to indicate that incompatible modules or features
    are selected/activated at the same time
    """

    pass
