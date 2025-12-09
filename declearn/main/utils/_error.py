"""Utils related to general errors"""


class IncompatibleConfigsError(Exception):
    """Custom exception to indicate that incompatible configs/modules/features
    are selected/activated at the same time
    """

    pass
