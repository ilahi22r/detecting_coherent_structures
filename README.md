# Detecting Coherent Structures via Dual E-tec

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

# [Add your basic usage example here]
```

## Method

The Dual E-tec algorithm:
1. Constructs a triangulation from Lagrangian trajectory data
2. Initializes material bands across the domain
3. Advects triangulation forward in time
4. Tracks edge crossings to compute topological invariants
5. Identifies coherent structures from crossing patterns and topological entropy

## Repository Structure
```
detecting_coherent_structures/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── src/                          # Core backend
│   ├── __init__.py
│   └── etec_dual.py             # Your triangulation backend (existing code)
│
├── examples/                     # Full implementation showcase
│   ├── __init__.py
│   ├── bickley_jet.py           # Complete analysis script
│   ├── velocity_fields.py       # Analytical velocity definitions
│   └── run_analysis.py          # Main executable
│
├── analysis/                     # Analysis tools (makes it impressive)
│   ├── __init__.py
│   ├── coherent_structures.py   # Structure detection algorithms
│   ├── topological_metrics.py   # Compute entropy, braid indices
│   └── comparison.py            # Compare with FTLE/traditional methods
│
├── visualization/                # Professional visualizations
│   ├── __init__.py
│   ├── plot_structures.py       # Plot detected structures
│   ├── plot_triangulation.py    # Visualize mesh evolution
│   ├── plot_trajectories.py     # Trajectory plots
│   └── animate.py               # Create animations
│
├── utils/                        # Utilities
│   ├── __init__.py
│   ├── integrate.py             # Trajectory integration
│   └── io.py                    # Save/load results
│
├── notebooks/                    # Jupyter demos (very impressive!)
│   └── bickley_jet_demo.ipynb   # Interactive demonstration
│
├── results/                      # Output directory
│   └── .gitkeep
│
└── docs/                         # Documentation (optional but impressive)
    ├── theory.md                # Mathematical background
    └── api.md                   # API documentation

```

## Requirements

- Python 3.7+
- NumPy
- SciPy
- Matplotlib

## License

MIT License

## Contact

ilahi22r@mtholyoke.edu
```

**Step 4: Create `requirements.txt`**
```
numpy>=1.20.0
scipy>=1.7.0
matplotlib>=3.3.0
```

**Step 5: Create `.gitignore`**
```
# Python
__pycache__/
*.py[cod]
*.so
*.egg-info/
dist/
build/

# Data and outputs
*.png
*.pdf
*.dat
*.h5

# IDEs
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db
