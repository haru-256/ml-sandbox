# ml_sandbox_libs

This repository contains a collection of utilities and libraries for machine learning experiments and development.

## Directory Structure

The repository is organized as follows:

```sh
ml-sandbox/
├── ml_sandbox_libs/       # Main package with utility libraries
│   ├── __init__.py        # Package initialization
│   ├── data/              # Data handling utilities
│   ├── models/            # Model implementations
│   ├── training/          # Training utilities
│   ├── evaluation/        # Metrics and evaluation tools
│   └── utils/             # General utilities
├── examples/              # Usage examples
├── tests/                 # Test cases
└── README.md              # This file
```

## Installation

### Local Development Installation

For local development, we recommend using [uv](https://github.com/astral-sh/uv) which provides fast package installation.

1. First, install uv if you don't have it:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create and activate a virtual environment:

```bash
# Create a virtual environment
uv venv

# Activate the virtual environment
source .venv/bin/activate  # On Unix/macOS
# OR
.venv\Scripts\activate     # On Windows
```

3. Install the package in development mode:

```bash
# Install the package locally in development mode
uv pip install -e .
```

### Installing with Dependencies

To install with all dependencies:

```bash
# Install with all dependencies
uv pip install -e ".[all]"
```

For specific dependency groups:

```bash
# Install with only torch-related dependencies
uv pip install -e ".[torch]"

# Install with only visualization dependencies
uv pip install -e ".[viz]"
```

## Usage

Import the libraries in your Python code:

```python
# Import modules from the package
from ml_sandbox_libs.utils import some_utility
from ml_sandbox_libs.models import some_model

# Use them in your code
result = some_utility.process_data(data)
model = some_model.create_model(params)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
