#!/bin/bash
pip install -r requirements-ci.txt
pytest tests/test_settings.py
