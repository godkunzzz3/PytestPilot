# Local Development Environment

FirstCoder supports Python 3.11 or newer. This environment was verified on Apple Silicon with Python 3.14.0, so there is no need to downgrade solely because Python 3.14 is newer. If ONNX Runtime or another native dependency has no compatible wheel on a different machine, use Python 3.12 and keep the old environment as a recoverable `.venv-*` directory.

## Install

Create and prepare the virtual environment with the repository's existing pip/editable-install workflow:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -e ".[dev,retrieval]"
.venv/bin/python -m pip check
```

The `retrieval` extra installs `qdrant-client[fastembed]`. Qdrant runs in local embedded mode for this MVP; Docker and a Qdrant server are not required. FastEmbed uses CPU/ONNX inference and the default MVP model is `BAAI/bge-small-en-v1.5`, which produces 384-dimensional vectors.

The model is downloaded from Hugging Face on first use. Keep its cache outside the repository, for example:

```bash
export FIRSTCODER_FASTEMBED_CACHE="$HOME/Library/Caches/firstcoder/fastembed"
```

Do not commit model files, virtual environments, Qdrant data, session data, or benchmark runs.

## Verify

Check the environment and installed dependencies:

```bash
.venv/bin/python --version
.venv/bin/python -m pip check
.venv/bin/python -c 'from fastembed import TextEmbedding; from qdrant_client import QdrantClient; print("retrieval imports ok")'
```

For a FastEmbed smoke check, initialize `TextEmbedding` with `BAAI/bge-small-en-v1.5`, the external cache directory, `cuda=False`, and `CPUExecutionProvider`; verify that both `query_embed` and `passage_embed` return finite 384-dimensional vectors. For Qdrant local, use `QdrantClient(path=<temporary-directory>)`, create a 384-dimensional COSINE collection, insert/query points, close it, and reopen the same path to verify persistence. Always remove the temporary database afterward.

Run the focused and complete test suites with the virtual-environment interpreter:

```bash
.venv/bin/python -m pytest \
  tests/test_local_pytest_benchmark.py \
  tests/test_eval_adapter.py -q

.venv/bin/python -m pytest tests -q
```

## Apple Silicon notes

- Use CPU packages only; do not install CUDA, `fastembed-gpu`, PyTorch, or TensorFlow for this MVP.
- If `onnxruntime` has no wheel for the active Python version, create a Python 3.12 environment instead of patching third-party packages.
- If Python's `urllib` reports `CERTIFICATE_VERIFY_FAILED` while downloading an EvalPlus fixture, point it at the CA bundle already installed in the environment: `export SSL_CERT_FILE="$(.venv/bin/python -c 'import certifi; print(certifi.where())')"`.
- An unauthenticated first model download can be rate-limited. Retry a finite number of times; do not put Hugging Face tokens or caches in the repository.
- A persistent Qdrant local path should be opened by one client process at a time. Close the client before reopening it.

While the Alibaba Bailian/Qwen quota is unavailable, do not run real Provider smoke tests, the default model-backed benchmark, or CLI flows that create a real Provider. Unit and integration tests must use Fake Providers or avoid Provider calls entirely. Never store API keys in project files or test output.
