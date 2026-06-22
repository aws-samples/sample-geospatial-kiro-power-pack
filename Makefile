# Geospatial Power Pack - developer tasks.
#
# The MCP SDK and most servers require Python >= 3.10. Override the interpreter
# used to build the venv with `make venv PYTHON=python3.13` if needed.

PYTHON ?= python3.12
VENV ?= .venv
VPY := $(VENV)/bin/python

# Servers exercised by the MCP handshake smoke test (no network). The "smoke"
# set is lightweight (no native build deps); the "heavy" set adds the servers
# whose wheels carry native libraries (GDAL/rasterio, shapely, pyproj, numpy).
# Override: `make smoke-mcp SERVERS="geo-index geo-ops"`.
SMOKE_SERVERS ?= geo-index geo-embedding-search
HEAVY_SERVERS ?= geo-ops geo-formats geo-raster geo-pointcloud geo-foundation-models geo-query
CREDENTIALED_SERVERS ?= geo-commercial-imagery geo-warehouse aws-geo-compute
SERVERS ?= $(SMOKE_SERVERS)

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help.
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*## "}{printf "  %-16s %s\n", $$1, $$2}'

.PHONY: venv
venv: ## Create the venv and install the MCP SDK, test tools, and geo-common.
	$(PYTHON) -m venv $(VENV)
	$(VPY) -m pip install --upgrade pip
	$(VPY) -m pip install mcp pytest pytest-asyncio hypothesis moto
	$(VPY) -m pip install -e ./packages/geo-common

.PHONY: install-smoke
install-smoke: ## Editable-install the lightweight servers used by smoke-mcp.
	$(VPY) -m pip install -e ./packages/geo-index -e ./packages/geo-embedding-search

.PHONY: install-heavy
install-heavy: ## Editable-install the native-dependency servers (GDAL/rasterio, shapely, pyproj, numpy).
	$(VPY) -m pip install $(addprefix -e ./packages/,$(HEAVY_SERVERS))

.PHONY: smoke-heavy
smoke-heavy: ## MCP handshake smoke for the native-dependency servers (run install-heavy first).
	$(VPY) scripts/smoke_mcp.py $(HEAVY_SERVERS)

.PHONY: install-credentialed
install-credentialed: ## Editable-install the credentialed/peer servers (lightweight, httpx-only).
	$(VPY) -m pip install $(addprefix -e ./packages/,$(CREDENTIALED_SERVERS))

.PHONY: smoke-credentialed
smoke-credentialed: ## Handshake + credential-guard smoke for credentialed servers (NO credentials needed).
	env -u SENTINELHUB_CLIENT_ID -u SENTINELHUB_CLIENT_SECRET \
		-u BIGQUERY_CREDENTIALS \
		-u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY -u AWS_REGION \
		$(VPY) scripts/smoke_mcp.py $(CREDENTIALED_SERVERS)

.PHONY: test
test: ## Run the full test suite (requires packages installed in the venv).
	$(VPY) -m pytest

.PHONY: smoke-import
smoke-import: ## Layer 1: import + construct every package (needs deps importable).
	$(PYTHON) scripts/smoke_import.py

.PHONY: smoke-mcp
smoke-mcp: ## Layer 2/3: MCP stdio handshake + local tool calls for $(SERVERS).
	$(VPY) scripts/smoke_mcp.py $(SERVERS)

.PHONY: smoke
smoke: smoke-import smoke-mcp ## Run the import and MCP-handshake smoke tests.

.PHONY: check-schemas
check-schemas: ## Validate every tool's MCP inputSchema (well-formed + no dangling $ref).
	$(VPY) scripts/check_tool_schemas.py

.PHONY: check-drift
check-drift: ## Check bundle-manifest.json agrees with server code (servers, install cmds, credentials, catalog).
	$(VPY) scripts/check_manifest_drift.py

.PHONY: check
check: check-schemas check-drift ## Run all static consistency checks (schemas + manifest drift).

.PHONY: new-server
new-server: ## Scaffold a new server package: make new-server NAME=geo-foo PILLAR=A TOOL=do_thing
	$(VPY) scripts/new_server.py --name "$(NAME)" --pillar "$(PILLAR)" --tool "$(TOOL)"

.PHONY: clean
clean: ## Remove caches and the venv.
	rm -rf $(VENV) .pytest_cache .hypothesis dist
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

.PHONY: build
build: ## Build a wheel for every package into dist/ (validates packaging metadata).
	PYTHON=$(VPY) scripts/build_wheels.sh

.PHONY: build-verify
build-verify: ## Build all wheels, then install + MCP-smoke a subset from the wheels.
	PYTHON=$(VPY) scripts/build_wheels.sh --verify-install
