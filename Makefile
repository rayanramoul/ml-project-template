SHELL := /bin/bash
PYTHON_VERSION = 3.10

# Variables that will be used mainly for the Docker build
PLATFORM = linux/arm64
EXAMPLE_DIR = examples
DOCKER_RUN_FLAGS = --env NEPTUNE_API_TOKEN $(NEPTUNE_API_TOKEN) --env NEPTUNE_PROJECT $(NEPTUNE_PROJECT) --privileged --network=host --ulimit nofile=65536:65536
DOCKER_RUN_FLAGS_GPU = $(DOCKER_RUN_FLAGS) --gpus all
export PROJECT_ROOT = $(shell pwd)

init:
	uv init -p $(PYTHON_VERSION)

install: configure_commit_template
	uv sync -p $(PYTHON_VERSION) && uv lock

# Configure git commit template, this will help to write better commit messages
configure-commit-template:
	git config --global commit.template $(realpath commit-template.txt)

# Pre-commit hooks are useful to run some checks (usually linters) before committing the code to the repository
# It will help to keep the code clean and consistent, this commands sets up the pre-commit hooks
configre-pre-commit:
	uvx pre-commit install

# Run pre-commit hooks on all files, it will run the checks on all files in the repository
format:
	uvx pre-commit run --all-files

# Run pytest to test the code unit tests under the 'tests/' directory
test:
	uv run pytest

# Use uv to run the train script while passing the arguments from the command line
train:
	uv run src/train.py ${ARGS}

# Use uv to run the evaluate script while passing the arguments from the command line
evaluate:
	uv run src/evaluate.py ${ARGS}

# Build the Docker image with the base dependencies
build-docker:
	docker build --target lightning-base -t lightning-base .

# Build the Docker image and jump into the container to test a fresh environment (CPU)
dev-container-cpu: build-docker
	docker run $(DOCKER_RUN_FLAGS) -v $(PROJECT_ROOT):/app -it lightning-base:latest /bin/bash

# Build the Docker image and jump into the container to test a fresh environment (GPU)
dev-container-gpu: build-docker
	docker run $(DOCKER_RUN_FLAGS_GPU) -v $(PROJECT_ROOT):/app -it lightning-base:latest /bin/bash

# Run the train script using the Docker image
train-docker: docker-build
	docker run $(DOCKER_RUN_FLAGS) --user root -v $(PROJECT_ROOT):/app lightning-base:latest /bin/bash -i -c "uv run /app/src/train.py ${ARGS}"

# Run the evaluate script using the Docker image
evaluate-docker: docker-build
	docker run $(DOCKER_RUN_FLAGS) --user root -v $(PROJECT_ROOT):/app lightning-base:latest /bin/bash -i -c "uv run /app/src/evaluate.py ${ARGS}"

# This build the documentation based on current code 'src/' and 'docs/' directories and deploy it to the gh-pages branch
# in your GitHub repository (you then need to setup the GitHub Pages to use the gh-pages branch)
deploy-pages:
	uv run mkdocs build && uv run mkdocs gh-deploy

# This is to run the documentation locally to see how it looks
serve-docs:
	uv run mkdocs build && uv run mkdocs serve
