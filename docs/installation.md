# Installation

## From PyPI (recommended)

```bash
pip install pybirewirex
```

Pre-built wheels are available for:

- **Linux** x86_64, aarch64 (manylinux)
- **macOS** x86_64, arm64
- **Windows** x86_64

## Optional extras

```bash
# igraph and networkx graph objects as inputs
pip install "pybirewirex[graph]"

# matplotlib + scikit-learn for scripts and visualisation
pip install "pybirewirex[vis]"

# everything (dev tools, docs, notebooks)
pip install "pybirewirex[dev]"
```

## From source

Requires a C compiler (GCC, Clang, or MSVC).

```bash
git clone https://github.com/lorenzoamir/PyBiRewire
cd PyBiRewire
pip install -e ".[dev]"
```

## Requirements

| Dependency | Minimum version |
|------------|----------------|
| Python | 3.10 |
| numpy | 1.23 |
| scipy | 1.9 |
| cffi | 1.0 |
