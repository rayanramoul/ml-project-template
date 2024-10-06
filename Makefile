SHELL := /bin/bash
PYTHON_VERSION = 3.10

# Some variables
EXAMPLE_DIR = examples


init:
	uv init -p $(PYTHON_VERSION)

install: configure_commit_template
	uv sync -p $(PYTHON_VERSION) && uv lock

configure_commit_template:
	git config --global commit.template $(realpath commit-template.txt)

format:
	uvx pre-commit run --all-files
