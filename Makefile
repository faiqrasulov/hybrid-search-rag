PYTHON ?= python
MEMCAP ?=

.PHONY: install data run test lint clean

install:
	$(PYTHON) -m pip install -e ".[dev]"

# Download BEIR SciFact (~2.7 MB) into data/ (cached).
data:
	$(PYTHON) -m hybridrag.cli download

# Full reproduction: index, embed (cached in data/embeddings), tune on train, evaluate on
# test, latency, bootstrap CIs -> reports/metrics.json + reports/figures/*.png.
run: data
	$(MEMCAP) $(PYTHON) -m hybridrag.evaluate --data-dir data --out reports

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .

clean:
	rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info
