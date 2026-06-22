# Contributing to the Geospatial Power Pack

This Power is a Python monorepo: each capability is a standalone, `uvx`-installable
MCP server under `packages/`, and every server depends on the shared
`geo-common` base. This guide covers how to add a new server, add a tool to an
existing server, the `BaseGeoServer` contract every server follows, the shared
error taxonomy, and the checks that keep the manifest and code in sync.

If you are extending the pack with a colleague's new MCP or a new tool, start
here.

---

## Prerequisites

Before scaffolding a server, set up a development environment:

- **Python 3.10 or higher** and **git**.
- **`make`** (for the helper targets) and the **`uv`** package manager
  (`curl -LsSf https://astral.sh/uv/install.sh | sh`).
- Clone the repository and change into it:
  ```bash
  git clone https://github.com/aws-samples/sample-geospatial-kiro-power-pack.git && cd sample-geospatial-kiro-power-pack
  ```
- Create and activate a virtual environment, then install the shared base:
  ```bash
  uv venv .venv && source .venv/bin/activate
  uv pip install -e ./packages/geo-common
  ```

## Quickstart: scaffold a new server

```bash
# Generate a working skeleton (server + model + tests + pyproject):
make new-server NAME=geo-foo PILLAR=A TOOL=do_thing
# or: python scripts/new_server.py --name geo-foo --pillar A --tool do_thing
```

This writes `packages/geo-foo/` with a `GeoFooServer` exposing a `do_thing`
tool, a result model, a Resource Catalog entry, an empty credential
declaration, and a passing test trio. Then:

1. **Register it for the schema check.** Add a row to the `SERVERS` tuple in
   `scripts/check_tool_schemas.py`:
   ```python
   ("geo_foo.server", "GeoFooServer"),
   ```
2. **Add it to the manifest.** Add a `servers` entry to `bundle-manifest.json`
   (name, pillar, status, tier, `uvx`, `role`, `credentials`, `wraps`). The
   `uvx` value must equal the server's `INSTALL_COMMAND`.
3. **Install it** into the dev venv:
   ```bash
   .venv/bin/python -m pip install -e ./packages/geo-foo
   ```
4. **Run the checks and tests:**
   ```bash
   make check        # schema + manifest-drift checks
   PYTHONPATH=packages/geo-common:packages/geo-foo .venv/bin/python -m pytest packages/geo-foo -q
   ```
5. **Replace the placeholder** tool/model/catalog with the real capability.

---

## The `BaseGeoServer` contract

Every server subclasses `geo_common.server.BaseGeoServer` and sets three class
attributes plus a constructor that registers its tools:

```python
class GeoFooServer(BaseGeoServer):
    pillar = "A"            # "A" (connectors), "B" (processing), "C" (GeoAI)
    server_name = "geo-foo" # matches the manifest name and the uvx command
    version = "0.1.0"

    def __init__(self, http=None) -> None:
        super().__init__(http=http)
        self.register_tool("do_thing", self.do_thing)
```

What the base gives you:

- **`register_tool(name, func)`** — registers an `async` method as an MCP tool.
  Its `inputSchema` is generated from the signature by
  `geo_common.runtime.build_tool_input_schema` (keyword-only params; pydantic
  model params get their `$defs` hoisted to the schema root automatically).
  Use **keyword-only** parameters (`def do_thing(self, *, value: str)`).
- **`self.http`** — a shared `geo_common.http.HttpClient` (retry/backoff,
  rate-limit handling, 30-second per-request timeout). Make all outbound calls
  through it so every server behaves identically; never create your own client.
- **`map_error(exc, source=...)`** — maps any library/transport exception onto
  exactly one taxonomy category. Wrap engine/driver calls so failures surface
  cleanly (an already-classified `GeoError` passes through unchanged).
- **`start(configured_keys=...)` / `run()`** — the startup credential guard and
  the stdio MCP serve loop. `main()` just calls `Server().run()`.

Two methods you should override:

- **`catalog_entries() -> list[CatalogEntry]`** — one entry per capability for
  the Resource Catalog. Each names `self.server_name` as `provider_server`, an
  `OpennessTier`, and a 1–500 char `capability_description`.
- **`required_credentials() -> list[CredentialSpec]`** — the `mcp.json` keys
  your server reads. Open servers return `[]`. These must agree exactly with
  the server's `credentials` block in `bundle-manifest.json` (the drift check
  enforces it).

---

## Error taxonomy

All errors are `geo_common.errors.GeoError` subclasses carrying exactly one of
seven categories. Raise the most specific one; never leak secret values in a
message or `detail`.

| Class | Category | Use for |
|---|---|---|
| `ValidationError` | `validation` | Malformed/out-of-range input. **Raise before doing any work**, naming the bad `parameter` in `detail`. |
| `AuthenticationError` | `authentication` | Missing/invalid credential. Name the `mcp.json` key, never the value. |
| `AuthorizationError` | `authorization` | Credential valid but action not permitted. |
| `RateLimitError` | `rate-limit` | 429 / quota (carries `retry_after`). |
| `NotFoundError` | `not-found` | Resource/job does not exist. |
| `UpstreamError` | `upstream` | Source 5xx or an unmapped external error. |
| `NetworkError` | `network` | Timeout / connection failure. |

Conventions:

- **Validate first.** Check inputs and raise `ValidationError` before any I/O,
  so a bad request never reaches an upstream.
- **Credential guards name the key, not the value.** An unconfigured
  credentialed source/engine raises `AuthenticationError` naming the `mcp.json`
  key (e.g. `NOAA_CDO_TOKEN`).
- **Optional credentials never block startup.** Classify keys
  `CredentialClassification.OPTIONAL` unless the whole server is useless
  without them; the startup guard then always lets the server start.
- **Re-tag availability errors with the logical source name** when wrapping
  several sources, so the error identifies *which* source was unavailable.

---

## Credentialed sources / engines

Open by default; credentials are additive. Patterns used across the pack:

- A **selectable** credentialed source (e.g. NOAA CDO `source="cdo"`, IUCN
  `source="iucn"`) is part of the source set so an unconfigured call returns a
  clean `AuthenticationError` naming the key. The source reads its token in
  `__init__` (the server passes `os.environ.get("KEY")`).

  > **Credential handling.** `mcp.json` `env` is the configuration surface, not
  > a secret store. Prefer short-lived credentials: an IAM role or an
  > `aws configure` profile for AWS, and a `chmod 600` file (referenced by path,
  > never committed) for file-based keys such as `BIGQUERY_CREDENTIALS`. Read a
  > key only by name, never log or echo its value, and surface a missing key as
  > an `AuthenticationError` that names the `mcp.json` key (not the value).
- A **fallback** credentialed source (e.g. Amazon Location) is only added to
  the set when its key is present, so the open default behaviour is unchanged
  when it is absent.
- When adding/removing a credential, update **both** the server's
  `required_credentials()` **and** the manifest `credentials` block, then run
  `make check-drift`.

---

## The required test trio (and more)

Every server ships at least three tests (the scaffold generates them):

1. **Registration/catalog** — the tool is registered, and each catalog entry
   names the server as provider at the right openness tier.
2. **Happy path** — a valid call returns the expected result shape.
3. **Validation guard** — a malformed input raises a `validation`-category
   error and does no work.

For sources that make HTTP calls, drive them through an
`httpx.MockTransport` wired into an `HttpClient` (see
`packages/*/tests/` for the `_client(handler)` helper pattern) so tests are
deterministic with no real network. Credentialed sources are tested the same
way: assert the request shape + auth header with a token, and assert the
missing-key path raises `AuthenticationError` naming the key. Do **not** add
tests that require real accounts or network for CI.

Run a single package's tests:

```bash
PYTHONPATH=packages/geo-common:packages/<pkg> .venv/bin/python -m pytest packages/<pkg> -q
```

Run everything from the repo root (packages installed in the venv):

```bash
.venv/bin/python -m pytest -q
```

---

## Static consistency checks

Run before opening a PR (`make check` runs both):

- **`make check-schemas`** — every tool's generated MCP `inputSchema` is a valid
  JSON Schema with no dangling `$ref` (the bug class that broke `spatial_join`).
- **`make check-drift`** — `bundle-manifest.json` and the server code agree:
  server set parity, `INSTALL_COMMAND` == manifest `uvx`, credential parity, and
  catalog integrity.

Both run with no network and no credentials.

---

## Adding a tool to an existing server

1. Add an `async def` keyword-only method and `self.register_tool("name", self.name)`
   in `__init__`.
2. Add a `CatalogEntry` for it in `catalog_entries()` (or extend the server's
   capability list).
3. Validate inputs first; raise `ValidationError` for bad input; wrap downstream
   failures with `map_error` or raise the specific `GeoError`.
4. Add tests (happy path + validation guard), and update any
   registration/catalog-set assertions that enumerate the server's tools.
5. `make check` and run the package tests.

---

## Conventions recap

- Python ≥ 3.10; `from __future__ import annotations` at the top of each module.
- Keyword-only tool parameters; pydantic models for structured results.
- One async `HttpClient` (the inherited `self.http`) for all network I/O.
- MIT-0 licensed; don't vendor third-party code — document companions instead
  (see the "Recommended companion MCPs" section in `README.md` / `POWER.md`).
- Keep the `README.md` Capabilities map, `bundle-manifest.json`, and
  `docs/testing-workflows.md` updated when you add a capability.

---

## Reporting bugs and feature requests

We welcome you to use the GitHub issue tracker to report bugs or suggest
features. When filing an issue, please check existing open and recently closed
issues to make sure somebody else hasn't already reported it. Try to include as
much information as you can — a reproducible test case or steps, the version
being used, and anything unusual about your environment or deployment.

## Contributing via pull requests

Contributions via pull requests are much appreciated. Before sending one, please
ensure that:

1. You are working against the latest source on the `main` branch.
2. You check existing open and recently merged pull requests to make sure
   someone else hasn't addressed the problem already.
3. You open an issue to discuss any significant work — we don't want your time
   wasted.

To send a pull request: fork the repository, modify the source focused on the
specific change, ensure local tests and `make check` pass, commit using clear
messages, and send the pull request answering any default questions in the
template. Pay attention to any automated CI failures and stay involved in the
conversation.

## Code of conduct

This project has adopted the [Amazon Open Source Code of Conduct](https://aws.github.io/code-of-conduct).
For more information see the [Code of Conduct FAQ](https://aws.github.io/code-of-conduct-faq) or contact
opensource-codeofconduct@amazon.com with any additional questions or comments.

## Security issue notifications

If you discover a potential security issue in this project we ask that you
notify AWS/Amazon Security via our
[vulnerability reporting page](https://aws.amazon.com/security/vulnerability-reporting/).
Please do **not** create a public GitHub issue.

## Licensing

See the [LICENSE](LICENSE) file for this project's licensing (MIT-0). We will ask
you to confirm the licensing of your contribution. Do not vendor third-party
code; document companions instead and keep [THIRD-PARTY-LICENSES](THIRD-PARTY-LICENSES)
accurate for any new dependency or referenced external.
