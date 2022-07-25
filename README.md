# Generic MIP framework

## Purpose
The aim of this repository is to...
* Unify the way MIP models are implemented in ECCO.
* Make models modular and highly customizable.
* Separate responsibilities by forcing the MIP implementations to separate constraints, objectives and variables.
* Separate responsibilities by removing settings logic from the model and let the model factory handle how to construct a model based on given settings.
* Abstract the solver layer to allow for fast switching between solver engines such as Google OR Tools, Gurobi and SCIP.

## Class overview
TODO

## Usage
TODO

## TODO
- [ ] Protect solver methods from bad inputs