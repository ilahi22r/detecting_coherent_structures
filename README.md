# Detecting Lagrangian Coherent Structures via Topological Advection

> Status: Work in Progess

> This repository is under active development and may change (11/30/2025)

A computational topology-based framework for detecting coherent structures in 2D time-dependent flows using topological analysis of Lagrangian data. A robust alternative to traditional LCS methods for sparse, noisy trajectory data.

## Overview

This repository implements the Dual E-tec method, which tracks how material curves (bands) cross edges in an evolving triangulation to identify transport barriers. Unlike traditional Lagrangian Coherent Structure (LCS) methods, this approach:

- Works with sparse, irregularly sampled trajectory data
- Does not require flow-field derivatives
- Uses topological invariants for robust structure detection
- Supports periodic boundary conditions (torus topology)

# Features

## Core Concepts

  Dual E-tec triangulation with evolving simplices
  
  Edge Crossing Counting to track material curve deformation
  
  Topological Entropy computation from crossing patterns
  
  Material Band Advection and curve stretching

## Geometry & Boundary Support

  Periodic boundaries (doubly periodic torus domains)
  
  Flexible triangulation updates
  
  Works with arbitrary time-dependent Lagrangian data

## Visualization Tools
  
  Particle trajectories
    
  Edge-crossing patterns
    
  Coherent structures
    
  Material band evolution

# Installation
```bash
git clone https://github.com/ilahi22r/detecting_coherent_structures.git
cd detecting_coherent_structures
pip install -r requirements.txt
```
Dependencies are minimal:
```
numpy
scipy
matplotlib
```


## Quick Start
Example import pattern:
```python
from src.etec_dual_periodic_bc_v2 import EtecDualPeriodicBCv2
from src.loop_combine import LoopCombine
```
For a full working example, see:
```
bickley_jet_example/prepare_traj_with_stationary.py
using_etec/detects_and_plots_coherentstructures.py
```

## Method Summary

The Dual E-tec algorithm proceeds as follows:

1. Construct an initial triangulation from Lagrangian trajectories.
2. Initialize topological bands across the triangulation.
3. Advect the triangulation forward in time using particle data.
4. Track band crossings over edges, building symbolic crossing matrices.
5. Compute topological invariants, including topological entropy.
6.Identify coherent structures as regions of low crossing complexity.

This method captures transport barriers without relying on Lyapunov exponents or spatial differentiation.

## Repository Structure
```
bickley_jet_example/        # Bickley Jet scripts & trajectory generation
using_etec/                 # Main coherent-structure detection routines
src/                        # Core implementation (Etec, LoopCombine)
README.md
requirements.txt
LICENSE
```

## License

Released under the MIT License.

## Contact
Rida Naveed Ilahi

ilahi22r@mtholyoke.edu

Mount Holyoke College



