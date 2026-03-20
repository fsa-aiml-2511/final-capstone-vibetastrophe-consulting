#!/bin/bash

# Use this script to install on MacOS.

uv venv --python 3.11 .venv
uv pip install -r requirements.txt

source .venv/bin/activate
