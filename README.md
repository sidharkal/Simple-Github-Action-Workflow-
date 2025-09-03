## Simple-Github-Action-Workflow

Iris Classification with GitHub Actions CI

This repository contains a simple Iris flower classification model built using Python (scikit-learn).
It also includes a GitHub Actions workflow that automatically trains the model and validates its accuracy on every push request.

# GitHub Actions Workflow

The workflow file is located at:
.github/workflows/iris-ci.yml

It runs automatically on:

Pushes to the main branch


# Workflow steps:

Checkout repository

Set up Python 3.10

Install dependencies (scikit-learn)

Run src/model.py
