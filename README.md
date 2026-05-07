# maxentbiomass

**A Python package for maximum‑entropy biomass models (Lurie–Wagensberg) with comparison of binning rules**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

## Overview

This package implements the **Lurie–Wagensberg maximum‑entropy biomass model**, which predicts an exponential distribution of individual mass under a single mean‑biomass constraint. It provides tools to:

- Choose the bin width (`Δm`) using **five different rules**:  
  Sturges, Scott, Freedman–Diaconis, Rice, and Knuth.
- Fit the exponential model to mass data.
- Evaluate model goodness‑of‑fit via **Kolmogorov–Smirnov test** and **AICc**.
- Compute diversity indices (Shannon index `k` and normalised diversity `μ̄`).

The package accompanies the manuscript:  
*“Objective discretisation in maximum‑entropy biomass models: a systematic comparison of binning rules and their effect on model selection and diversity inference”* (Sira et al., in review, *Methods in Ecology and Evolution*).

## Installation

You can install the package directly from GitHub:

```bash
pip install git+https://github.com/neuroq33/maxentbiomass.git
