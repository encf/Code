# Mango

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Version](https://img.shields.io/badge/version-1.0.0-brightgreen.svg)
![Build](https://img.shields.io/badge/build-passing-success.svg)

## Description

Mango is the implementation of the protocol presented in our paper. It includes tools to verify the correctness of the protocol and benchmark its execution speed. This repository contains two main components:

- **correctness.py**: A Python script to validate the correctness of the protocol.
- **benchmark.cpp**: A C++ program to measure the execution speed of the protocol. Note: This file is solely for benchmarking purposes and may contain code errors.

The project leverages external libraries such as [GMP](https://gmplib.org/) and [emp-toolkit](https://github.com/emp-toolkit/emp-tool) for efficient computation.

## Features

- Protocol correctness verification using Python.
- Speed benchmarking of the protocol using C++.
- Utilization of GMP and emp-toolkit for computation.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Correctness Verification](#correctness-verification)
  - [Benchmarking](#benchmarking)
- [License](#license)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/encf/Mango.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Mango
   ```

### Dependencies
- **Operate System**: 22.04.1-Ubuntu
- **correctness.py**: Requires Python 3.x. 

- **benchmark.cpp**: Requires a C++ compiler and the [GMP](https://gmplib.org/) and [emp-toolkit](https://github.com/emp-toolkit/emp-tool) libraries. Ensure they are installed and properly configured on your system.

## Usage

### Correctness Verification

Run the `correctness.py` script to validate the protocol:
```bash
python correctness.py
```
This script verifies the correctness of the protocol implementation based on the specifications in the paper.

### Benchmarking

Compile and execute the `benchmark.cpp` file to test the execution speed of the protocol:
```bash
cmake CMakeLists.txt
make
./Main
```
**Note**: The `benchmark.cpp` file is intended only for speed testing and may contain errors.

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/encf/Mango/blob/main/LICENSE) file for details.
