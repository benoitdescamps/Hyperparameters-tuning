'''
Meta-implementations of the base-class Base.SHBaseEstimator .
current support:
    * sklearn: Pipeline (Assume model is at the end, and belongs to this list)
    * XGBRegressor
    * XGBClassifier
'''

import os
if not os.path.exists('__cache__'):
    os.makedirs('__cache__')
