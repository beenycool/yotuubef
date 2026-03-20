#!/bin/bash
set -e
pip install -r requirements-ci.txt
pytest
