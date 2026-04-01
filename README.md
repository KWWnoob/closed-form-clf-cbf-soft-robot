# A Closed-Form CLF-CBF Controller for Whole-Body Continuum Soft Robot Collision Avoidance

Official implementation of the paper:

**A Closed-Form CLF-CBF Controller for Whole-Body Continuum Soft Robot Collision Avoidance**  
Accepted at **IEEE RoboSoft 2026**

---

## Overview

This repository contains simulation code for **setpoint tracking with obstacle avoidance** on a **tendon-actuated piecewise constant strain (PCS) soft robot**, using a **CLF-CBF-based controller**.

The implementation uses:
- **JAX** for differentiable computation
- **Diffrax** for ODE simulation
- **soromox** for soft robot modeling
- a **modified local version of `cbfpy`** for CLF-CBF control

Main features:
- End-effector **setpoint tracking**
- Obstacle avoidance using a **Control Barrier Function (CBF)**
- Goal convergence using a **Control Lyapunov Function (CLF)**
- Time-domain simulation with **Diffrax**
- Visualization with **Open3D**
- CSV export of robot states

---

## Modified Dependency: `cbfpy`

This project uses a modified version of `cbfpy`, included under `third_party/cbfpy`.

The original `cbfpy` library was developed by its respective authors and is available at:  
[https://github.com/danielpmorton/cbfpy](https://github.com/danielpmorton/cbfpy)

We gratefully acknowledge the original authors for making their implementation publicly available.

Modifications in this repository include:
- extension of `cbfpy` with the **closed-form CLF-CBF controller** proposed in our paper (implemented in `clf_cbf.py`)
- adaptations for **tendon-actuated soft robot control**
- integration with the **CLF-CBF simulation pipeline** used in this repository

Any errors or modifications introduced are the responsibility of this repository.

---

## Project Structure

```bash
.
├── README.md
├── requirements.txt
├── third_party/
│   └── cbfpy/                  # modified local version of cbfpy
├── viz/
│   └── open3d_vis_V2.py
├── examples/
│   └── simulation_kinematics.py
└── ...
```

---

## Installation

We recommend using a fresh Python environment.

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### 2. Create and activate a Python environment

Using `venv`:

```bash
python -m venv venv
source venv/bin/activate
```

Or using Conda:

```bash
conda create -n softrobot python=3.10
conda activate softrobot
```

### 3. Install core dependencies

```bash
pip install -r requirements.txt
```

### 4. Local third-party dependency

This repository includes a modified local version of `cbfpy` under:

```bash
third_party/cbfpy
```

No separate installation of `cbfpy` is required.

The example scripts are configured to use the local modified version included in this repository.

### 5. Notes on external dependencies

This project additionally depends on:
- `soromox`
- `open3d`

If `soromox` is not available via `pip` in your environment, please install it manually from its official source.

### 6. JAX installation note

This project uses JAX in **64-bit mode**.

By default, the CPU version is sufficient for running the provided examples.  
If GPU support is desired, please follow the official JAX installation instructions:

https://github.com/google/jax#installation

---

## Running the Example

Run the kinematic simulation example:

```bash
python examples/simulation_kinematics.py
```

This script will:
- build a 2-segment tendon-actuated soft robot
- define the CLF-CBF controller
- simulate the closed-loop system
- plot:
  - minimum obstacle clearance over time
  - end-effector trajectories over time
- visualize the robot motion in Open3D
- save the state trajectory to a CSV file

---

## Output

The simulation exports the robot state trajectory as a CSV file, e.g.:

```bash
setpoint_results_new.csv
```

This can be used for:
- post-processing
- plotting
- debugging
- controller evaluation

---

## Notes

### On `soromox` compatibility
This code is written to be compatible with newer `soromox` APIs, while also including fallback behavior for older interfaces when possible.

### On reproducibility
For reproducible results, please use:
- the included local modified version of `cbfpy`
- a compatible version of `soromox`
- JAX with 64-bit mode enabled

### On local package resolution
The example scripts are expected to use the local `third_party/cbfpy` version rather than an externally installed version.

---

## Citation

If you find this repository useful, please consider citing:

```bibtex
@inproceedings{wong2026closedform,
  title={A Closed-Form CLF-CBF Controller for Whole-Body Continuum Soft Robot Collision Avoidance},
  author={Wong, Kiwan and Stölzle, Maximillian and Xiao, Wei and Rus, Daniela},
  booktitle={IEEE International Conference on Soft Robotics (RoboSoft)},
  year={2026}
}
```

---

## License

This repository is released under the license specified in the `LICENSE` file.

Please also refer to the original license of `cbfpy` under `third_party/cbfpy`.
