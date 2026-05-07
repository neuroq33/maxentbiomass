# maxentbiomass

**Python package for maximum-entropy biomass models (Lurie–Wagensberg) with comparison of binning rules**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)  **(Actualiza el DOI cuando lo tengas)*

## Overview

This package implements the Lurie–Wagensberg maximum‑entropy biomass model, which predicts an exponential distribution of individual mass under a mean‑biomass constraint. It provides functions to:
- Choose the bin width (`Δm`) using five different rules: Sturges, Scott, Freedman–Diaconis, Rice, and Knuth.
- Fit the exponential model to mass data.
- Evaluate model fit via Kolmogorov–Smirnov test and AICc.
- Compute diversity indices (Shannon index `k` and normalised diversity `μ̄`).

## Installation

You can install the package directly from GitHub:

```bash
pip install git+https://github.com/neuroq33/maxentbiomass.git
