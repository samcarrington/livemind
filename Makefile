-include .env
export

RG      ?= $(AZURE_RESOURCE_GROUP)
ACR      ?= $(ACR_NAME)
ACR      ?= inventaiacr
IMAGE    ?= livemind
TAG      ?= latest
REGISTRY  = $(ACR).azurecr.io
FULL_TAG  = $(REGISTRY)/$(IMAGE):$(TAG)

.PHONY: login build push deploy stt run

login:
	az acr login -n $(ACR) -g $(RG)

build:
	docker build -t $(FULL_TAG) .

push:
	docker push $(FULL_TAG)

deploy: login build push

# ── Local (no tunnel): Ollama + local STT on this machine ──
stt:
	uv run --extra stt-mlx --extra stt python stt_server.py

run:
	uv run python app.py --host 0.0.0.0 --port 8765
