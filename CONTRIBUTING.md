# Contributing

## Quick Start

```bash
uv venv # install if you don't already
source .venv/bin/activate
uv sync
uv run pre-commit install
```

## Before Committing

```bash
uv run ruff check . --fix
uv run ruff format .
```

## Adding Features

1. Fork/branch
2. Make changes
3. Test it works (this codebase has no tests, sorry): `uv run train.py training=fast`
    - just ensure it doesn't break, do some simple sanity checks pls.
4. PR with a short description

## Questions?

Email me: abs6bd@virginia.edu
