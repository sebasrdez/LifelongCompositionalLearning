# Compositional Lifelong Learning

This is the source code used for [Aprendizaxe permanente en contornas dinámicas: enfoques compositivos para a robótica. (Sebastián Rodríguez Trillo, 2023)]. 

This package contains the implementations of the algorithms that conform to our framework for compositional learning: ER, EWC, and NFT with linear models combinations, soft layer ordering, and soft gating. Deep learning variants are implemented with and without dynamic addition of new modules. Implementations for all baselines in the paper are also included: jointly trained and no-components.

## Installation

All dependencies are listed in the `env.yml` file (Linux). To install, create a Conda environment with:
 
```$ conda env create -f env.yml```

Activate the environment with:

```$ conda activate lifelongcompositionallearning```

You should be able to run any of the experiment scripts at that point.

## Code structure

The code structure is the following:

* `experiment_script_soft.py` -- Script for running multiple experiments with soft layer ordering
* `experiment_script_gated.py` -- Script for running multiple experiments with soft gating 
* `experiment_script_linear.py` -- Script for running multiple experiments with linear model combinations 
* `lifelong_experiment.py` -- Experiment script for running training with one configuration.
* `datasets/`
    * `DATASET_NAME` -- each directory contains the raw data, the processing code we used (where applicable), and the processed data. 
    * `datasets.py` -- a Python class wrapper for each data set, which creates a `torch.utils.data.TensorDataset` for each task in the data set.
* `learners/`
    * `base_learning_classes.py` -- Base classes for compositional, dynamic + compositional, joint, and no-components agents.
    * `*_compositional.py` -- Compositional agents
    * `*_dynamic.py` -- Dynamic + compositional agents
    * `*_joint.py` -- Jointly trained agents
    * `*_nocomponents.py` -- No-components agents
    * `er_*.py` -- ER-based agents
    * `ewc_*.py` -- EWC-based agents
    * `nft_*.py` -- NFT-based agents
* `models/`
    * `base_net_classes.py` -- Base classes for deep compositional models
    * `linear.py` -- No-components linear model
    * `linear_factored.py` - Factored linear model
    * `mlp*.py` -- Fully-connected nets
    * `*_soft_lifelong.py` -- Soft layer ordering
    * `*_soft_lifelong_dynamic.py` -- Soft layer ordering supporting dynamic component additions
    * `*_soft_gated_lifelong.py` -- Soft gating net
    * `*_soft_gated_lifelong_dynamic` -- Soft gating net supporting dynamic component additions
* `utils/`
    * `kfac_ewc.py` -- Kronecker-factored Hessian approximator for EWC, implemented as a PyTorch optimizer/preconditioner
    * `bars_loss.py` -- Code for creating the bar chart to compare the different methods on the test set
    * `loss_lifelong_plots.py` -- Code for creating the loss evolution graphs throughout the training in each task
    * `replay_buffers.py` -- Implementation of the replay buffers for ER-based algorithms as a `torch.utils.data.TensorDataset`
* `results/`
    * `results_linear/` -- Results of linear model combinations method
    * `results_soft/` -- Results of soft layer ordering method
    * `results_gated/` -- Results of soft gating method