SHELL := /bin/bash
PYTHON_VERSION = 3.10

# Some variables
PLATFORM = linux/arm64
EXAMPLE_DIR = examples
DOCKER_RUN_FLAGS = --env NEPTUNE_API_TOKEN $(NEPTUNE_API_TOKEN) --env NEPTUNE_PROJECT $(NEPTUNE_PROJECT) --env NEPTUNE_EXPERIMENT $(NEPTUNE_EXPERIMENT) --platform=linux/amd64 --privileged  --network=host --ulimit nofile=65536:65536 
DOCKER_RUN_FLAGS_GPU = $(DOCKER_RUN_FLAGS) --gpus all


init:
	uv init -p $(PYTHON_VERSION)

install: configure_commit_template
	uv sync -p $(PYTHON_VERSION) && uv lock

configure_commit_template:
	git config --global commit.template $(realpath commit-template.txt)

format:
	uvx pre-commit run --all-files

train:
	uv run src/train.py ${ARGS}

evaluate:
	uv run src/evaluate.py ${ARGS}

docker-build:
	docker build --platform=${PLATFORM} --target lightning-base -t lightning-base .

train-docker: docker-build
	docker run lightning-base:latest  /bin/bash -i -c "uv run src/train.py ${ARGS}"

evaluate-docker: docker-build
	docker run lightning-base:latest  /bin/bash -i -c "uv run src/evaluate.py ${ARGS}"
