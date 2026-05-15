# Hybrid-BBB-LNP-Model

Hybrid ODE-agent-based framework for predicting BBB delivery of CRISPR-loaded lipid nanoparticles.

## Overview

This repository contains a hybrid multiscale computational framework combining:
- compartmental ordinary differential equation (ODE) modeling
- agent-based simulation (ABS)

for predicting blood–brain barrier (BBB) delivery behavior of lipid nanoparticles (LNPs).

The framework models:
- systemic nanoparticle transport
- BBB permeability
- hepatic sequestration
- stochastic particle-level uptake behavior
- nanoparticle heterogeneity

## Repository Structure

scripts/
    hybrid_bbb_model.py

## Requirements

Python 3.10+

Required packages:
- numpy
- scipy
- matplotlib
- pandas

## Installation

pip install -r requirements.txt

## Usage

Run the main simulation:

python scripts/hybrid_bbb_model.py

## Citation

If you use this framework, please cite the associated manuscript.

## License

MIT License
