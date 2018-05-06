'''
Meta-implementations of _run_then_return_val_loss .
current support:
    * sklearn: Pipeline (Assume model is at the end, and belongs to this list)
    * XGBRegressor
    * XGBClassifier
'''