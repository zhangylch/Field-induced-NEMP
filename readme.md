# NEMP
## Introduction:
This repository provides an implementation of Field-induced Node-Equivariant Message Passing Neural Networks (NEMP) for representing field-induced potential energy surfaces, built on the JAX framework. 

Molecular dynamics (MD) simulations can be performed using the JAX-MD package to fully leverage GPU acceleration and achieve optimal performance. For more details about the methodology and benchmarks, please refer to the associated paper.
## Requirements:
* jax+flax
* optax
* ASE
* JAX-MD

## Examples:
Training can be performed by running the following command:
```
python3 $path "train"
```
where $path is the directory where the code is located. This command executes the train.py script inside the train folder.

All training parameters are specified in the config.json file, and the accepted input data format is extended XYZ. Examples of config.json are provided in the examples folder.

## References
If you use this package, please cite these works.
1. NEMP model: Yaolong Zhang and Hua Guo. [abs/2508.16086](https://arxiv.org/abs/2508.16086)
