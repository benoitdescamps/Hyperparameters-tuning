
class EarlyStopException(Exception):
    """Pattern from XGGBOOST Python API (https://github.com/dmlc/xgboost)
    Exception to signal early stopping.
    Parameters
    ----------
    None
    """
    def __init__(self):
        super(EarlyStopException, self).__init__()