# Directory Documentation

## Overview
This directory contains utilities for processing and converting model logs into structured data formats.

## Files

### `layer_convert.py`
Converts model logs from JSON format to Parquet format for improved data storage and query performance.

## Setup

Before using the utilities in this directory, install the required dependencies:
```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n moe-env python
conda activate moe-env

# Or using uv
uv venv
source .venv/bin/activate

# Install dependencies
pip install -r pipelines/requirements.txt
```