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
./example
```

## Usage

The main components include:

### Core Data Types (`include/data_types.h`)
- `CscMatrix`: Compressed sparse column matrix format
- `QDLDLData`: Data structure for QDLDL factorization
- Type aliases for Eigen matrices and vectors

### KKT System Formation (`include/utils.h`)
The `form_KKT_system` function constructs the KKT matrix and the right-hand side vector.

### QDLDL Interface (`include/qdldl_interface.h`)
- `qdldl_setup`: Initialize QDLDL data structures
- `qdldl_cleanup`: Free allocated memory

## Example: Quadrotor Control

The example (`example.cpp`) demonstrates LQR control for a quadrotor with:

- **State dimension**: 12 (position, orientation, velocities)
- **Input dimension**: 4 (motor thrusts)

The system dynamics are linearized around the hover condition. The problem data can be found [here](https://osqp.org/docs/release-0.6.3/examples/mpc.html).

## References

- [QDLDL: A Free LDL Factorisation Routine](https://github.com/osqp/qdldl)
- Stellato, B., Banjac, G., Goulart, P., Bemporad, A., & Boyd, S. (2020). OSQP: An operator splitting solver for quadratic programs. Mathematical Programming Computation, 12(4), 637-672.
