# Batter Dashboard (Data Layer Scaffold)

This project is the foundational data layer for an MLB hitter game-level performance dashboard.

## Install

```bash
pip install -e .
```

## Smoke tests

```bash
python -m data.loader
python -m data.run_value_matrix
```

## Run tests

```bash
pytest
```

## Cache

Parquet cache files are written to `cache/` under the project root.

To clear cache:

```bash
rm -f cache/*.parquet
```
