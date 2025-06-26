# QDLDL-LQR

A C++ implementation of Linear Quadratic Regulator (LQR) control using the QDLDL sparse LDL factorization routine for solving the associated KKT (Karush-Kuhn-Tucker) system.

## Overview

This project demonstrates how to solve LQR problems using sparse linear algebra. The implementation:

- Formulates the LQR problem as a KKT system
- Uses the QDLDL library for efficient sparse LDL factorization

For more information, please refer to the [tutorial](https://luyao787.github.io/blog/2025/LQP2/).

## Dependencies

- **CMake** (>= 3.5.2)
- **Eigen3** - Linear algebra library
- **QDLDL** - Sparse LDL factorization library

## Building

1. Clone the repository:
```bash
git clone https://github.com/Luyao787/QDLDL-LQR.git
cd QDLDL-LQR
```

2. Clone the QDLDL library:
```bash
git clone https://github.com/osqp/qdldl.git
```

3. Create a build directory and compile:
```bash
mkdir build
cd build
cmake ..
cmake --build .
```

4. Run the example:
```bash
./example --banded
```
This runs the LQR solver using a banded KKT matrix approach with decision variables ordered as: `[u_0, λ_1, x_1, u_1, λ_2, x_2, ..., u_{N-1}, λ_N, x_N]`.

```bash
./example --qp
```
This runs the LQR solver formulated as a quadratic programming problem with decision variables ordered as: `[u_0, x_1, u_1, x_2, ..., u_{N-1}, x_N, λ_1, λ_2, ..., λ_N]`.


## Usage

The main components include:

### Core Data Types (`include/data_types.h`)
- `CscMatrix`: Compressed sparse column matrix format
- `QDLDLData`: Data structure for QDLDL factorization
- Type aliases for Eigen matrices and vectors

### KKT System Formation (`include/utils.h`)
The implementation provides two KKT system formulations:

- **Banded KKT System**: Uses `form_banded_KKT_system` with primal and dual variables ordered by time step
- **QP-type KKT System**: Uses `form_QP_KKT_system` with primal variables grouped together

Both approaches construct the KKT matrix and right-hand side vector for the LQR problem.

### Decision Variable Ordering

#### Banded Approach (`--banded`)
Variables are ordered as: `[u_0, λ_1, x_1, u_1, λ_2, x_2, ..., u_{N-1}, λ_N, x_N]`

Where:
- `u_k`: Control input (dimension: 4 for quadrotor)
- `λ_k`: Lagrange multiplier (dimension: 12 for quadrotor)  
- `x_k`: State (dimension: 12 for quadrotor)

#### QP Approach (`--qp`)
Variables are ordered as: `[u_0, x_1, u_1, x_2, ..., u_{N-1}, x_N, λ_1, λ_2, ..., λ_N]`

This separates primal variables (controls and states) from dual variables (Lagrange multipliers).

### QDLDL Interface (`include/qdldl_interface.h`)
- `qdldl_setup`: Initialize QDLDL data structures
- `qdldl_cleanup`: Free allocated memory

## Example: Quadrotor Control

The example (`example.cpp`) demonstrates LQR control for a quadrotor with:

- **State dimension**: 12 (position, orientation, velocities)
- **Input dimension**: 4 (motor thrusts)

The system dynamics are linearized around the hovering condition. The problem data can be found [here](https://osqp.org/docs/release-0.6.3/examples/mpc.html).

## References

- [QDLDL: A Free LDL Factorisation Routine](https://github.com/osqp/qdldl)
- Stellato, B., Banjac, G., Goulart, P., Bemporad, A., & Boyd, S. (2020). OSQP: An operator splitting solver for quadratic programs. Mathematical Programming Computation, 12(4), 637-672.
