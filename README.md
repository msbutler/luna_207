# Learned Uncertainty-Aware (LUNA) Bases for Bayesian Regression using Multi-Headed Auxiliary Networks

# AM207 Fall 2020 Final Project

#### Contributors: Michael Butler, Max Cembalest, M. Elaine Cunha

The original LUNA paper can be found [here](https://arxiv.org/abs/2006.11695).

### Final Report

---


* `FinalReport.ipynb` - summary of the original LUNA paper, evaluation of experimental design, and remarks on our process of replicating the authors' results

### Neural Network and LUNA Implementations

---

* `feed_forward.py` - Contains the base class for a neural network, adapted from in-class code
* `nlm.py` - Contains the base class for the NLM model which defines a neural linear model
* `luna.py` - Contains the base class for the LUNA model; inherits from `nlm.py`
* `bayes_helpers.py` - Helper functions for Bayesian analysis within the LUNA model; contains functions for sampling from the prior or posterior, calculating the prior/posterior predictive, plotting predictive intervals, calculating
* `utils.py` - helper functions for neural network components of the LUNA model; contains functions for generating training data (with a gap) and running a toy neural net/plotting results
* `config.py` - Contains standardized configuration parameters for NLM and LUNA models

### Additional NLM and LUNA Jupyter Notebook Demos

---

* `LUNABaseDemo.ipynb` - demonstration of LUNA model on a toy dataset
* `PriorPredictives_Demo.ipynb` - demonstration of how regularization affects the prior and posterior predictive of an NLM
* `CategoricalFailure.ipynb` - demonstration of a LUNA failure mode: uncertainty estimation for 2-D classification
### Informative Plots and Data Files
---

* Directory **figs_final** - Contains plots of true data, predictions, and predictive uncertainty for NLM and LUNA training examples.