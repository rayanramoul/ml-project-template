SHELL := /bin/bash
PYTHON_VERSION = 3.10

# Some variables
EXAMPLE_DIR = examples


init:
	uv init -p $(PYTHON_VERSION)

configure_commit_template:
	git config --global commit.template $(realpath commit-template.txt)

