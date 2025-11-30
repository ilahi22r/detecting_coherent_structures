# Detecting Lagrangian Coherent Structures via Topological Advection

A computational topology-based framework for detecting coherent structures in 2D time-dependent flows using topological analysis of Lagrangian data. A robust alternative to traditional LCS methods for sparse, noisy trajectory data.

## Overview

This repository implements the Dual E-tec method, which tracks how material curves (bands) cross edges in an evolving triangulation to identify transport barriers. Unlike traditional Lagrangian Coherent Structure (LCS) methods, this approach:

- Works with sparse, irregularly sampled trajectory data
- Does not require flow-field derivatives
- Uses topological invariants for robust structure detection
- Supports periodic boundary conditions (torus topology)

## Features

- **Dual E-tec Algorithm**: Triangulation-based edge tracking
- **Topological Entropy**: Compute entropy from crossing patterns
- **Material Curve Advection**: Track curve evolution in unsteady flows
- **Periodic Boundaries**: Handle flows on torus (doubly periodic domains)
- **Visualization**: Built-in plotting functionality

## Installation
```bash
git clone https://github.com/ilahi22r/detecting_coherent_structures.git
cd detecting_coherent_structures
pip install -r requirements.txt
```

## Quick Start
```python
from src.etec_dual import EtecDualPeriodicBC
```

## Method

The Dual E-tec algorithm:
1. Constructs a triangulation from Lagrangian trajectory data
2. Initializes material bands across the domain
3. Advects triangulation forward in time
4. Tracks edge crossings to compute topological invariants
5. Identifies coherent structures from crossing patterns and topological entropy


## License

MIT License

## Contact

ilahi22r@mtholyoke.edu



