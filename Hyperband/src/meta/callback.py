# additional callback functions for update
from .core import EarlyStopException
import numpy as np
def early_stop(stopping_rounds=10, maximize=True):
    def callback(env,score,iteration):
        if score >= env['best_iteration']:
            env['best_score'] = score
            env['best_iteration'] = iteration
        elif (iteration - env['best_iteration'])> stopping_rounds:
            raise EarlyStopException()

    return callback

