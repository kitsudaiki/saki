# Saki

![Latest Release](https://img.shields.io/github/v/release/kitsudaiki/saki?include_prereleases&label=Version&style=flat-square)
![License](https://img.shields.io/github/license/kitsudaiki/saki?style=flat-square)
![Platform](https://img.shields.io/badge/Platform-Linux-blue?style=flat-square)
![Architecture](https://img.shields.io/badge/Architecture-amd64%20%2B%20arm64-blue?style=flat-square)

[![Github workflow status](https://img.shields.io/github/actions/workflow/status/kitsudaiki/saki/build_test.yml?branch=develop&style=flat-square&label=Build%20and%20Test)](https://github.com/kitsudaiki/saki/actions/workflows/build_test.yml)
[![RS Report](https://rust-reportcard.xuri.me/badge/github.com/kitsudaiki/saki?style=flat-square)](https://rust-reportcard.xuri.me/report/github.com/kitsudaiki/saki)
[![CodeQL](https://img.shields.io/github/actions/workflow/status/kitsudaiki/saki/codeql.yml?branch=develop&style=flat-square&label=CodeQL)](https://github.com/kitsudaiki/saki/actions/workflows/codeql.yml)
[![OpenSSF Scorecard](https://img.shields.io/ossf-scorecard/github.com/kitsudaiki/saki?branch=develop&style=flat-square&label=OpenSSF-Scorecard)](https://scorecard.dev/viewer/?uri=github.com/kitsudaiki/saki)

<p align="center">
  <img src="assets/logo.jpg" width="500" height="500" />
</p>

**This repository contains the core of the previous project [Ainari](https://github.com/kitsudaiki/ainari)**

Saki contains in its core a custom experimental artificial neural network, which can work on
unnormalized and unfiltered input-data, like sensor measurement data. The network growth over time
by creating new nodes and connections between the nodes while learning new data. The original
concept was created by myself, merged with classical deep-learning and the code was written from
scratch without any frameworks. The goal behind Saki is to create something unique, which works
more like the human brain. It wasn't targeted to get a higher accuracy than classical artificial
neural networks like Tensorflow, but to be more flexible and easier to use and more efficient in
resource-consumption for big amounts of inputs and users. Additionally it also provides an
as-a-Service architecture within a cloud native environment and multi-tenancy.

## Current experimal and prototypically implemented features:

- **Growing neural network**:

    The artificial neural network, which is the core of the project, growth over time while learning
    new things by creating new nodes and connections between the nodes based on the given input. A
    resize of the network is also quite linear in complexity.

- **No normalization of input**

    The input of the network is not restricted to range of 0.0 - 1.0 . Every value can be inserted,
    even negative values. Also if there is a single broken value in the input-data, which is million
    times higher, than the rest of the input-values, it has nearly no effect on the rest of the
    already trained data.

- **No strict layer structure**

    The base of a new neural network is defined by a cluster-template. In these templates the
    structure of the network in planed in hexagons, indeed of layer. When a node tries to create a
    new synapse, the location of the target-node depends on the location of the source-node within
    these hexagons. The target is random and the probability depends on the distance to the source.
    This way it is possible to break the static layer structure. But when defining a line of
    hexagons and allow nodes only to connect to the nodes of the next hexagon, a classical
    layer-structure can still be enforced.

- **Spiking neural network**

    The concept also supports a special version of working as a spiking neural network. This is
    optional for a created network and basically has the result, that an input is impacted by an
    older input, based on the time how long ago this input happened.

- **3-dimensional networks**

    It is basically possible to define 3-dimensional networks. This was only added, because the human
    brain is also a 3D-object. This feature exist, but was
    never tested until now. Maybe in bigger tests in the future this feature could become useful to
    better mix information with each other.

- **Rust as programming language for the backend without unsafe**

    Even the project started with C++ as primary programming language until v0.7.0, the whole backend
    is now written in Rust without unsafe code and use `#![forbid(unsafe_code)]` to prevent the
    usage of unsafe. Based on `cargo geiger` many used dependencies sadly still use much unsafe
    code, but at least in this repository here no unsafe code is added.

- **Parallelism**

    The processing structure works also for multiple threads, which can work at the same time on the
    same network. (GPU-support with CUDA is disabled at the moment for various reasons).


## How to build

- Create and activate a virtual environment

  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

- Install maturin

  ```bash
  pip3 install maturin
  ```

- Build and install the library locally into the virtual environment

  ```bash
  maturin develop
  ```
