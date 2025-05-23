#!/bin/bash

# Install frontend dependencies
npm --prefix frontend install

# Install backend dependencies
pip install -r backend/requirements.txt
