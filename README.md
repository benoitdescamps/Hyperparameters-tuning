# Hyperparameters-tuning
Series of notes/code-snippets about various hyperparameter optimisation techniques.

## Allocation-based optimisations
### SuccessiveHalving
Code is based on the paper by Talwalkar A. [Non-stochastic best arm identiﬁcation and hyperparameter optimization](https://arxiv.org/abs/1502.07943).
Some Example code for :
* sklearn: RandomForest, GradientBoosting, etc...
* xgboost

More information can be found in [this blog post](https://medium.com/machine-learning-rambling/tuning-hyperparameters-part-i-successivehalving-c6c602865619)
### Hyperband
Code is based on the paper by Talwalkar A. [Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization](https://arxiv.org/abs/1603.06560).

### TODO:
* add setup.py
* add example Extreme-random-forest
* add proper hyperparameter-spaces in examples
* ? comparison for some dataset?
### References:
* [zygmuntz hyperband](https://github.com/zygmuntz/hyperband0)

## Random Search
Spark implementation of Random  Search for sampling over breeze.stats.distribution.
More information can be found in [this blog post](https://medium.com/machine-learning-rambling/hyperparameters-part-ii-random-search-on-spark-77667e68b606)