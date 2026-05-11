.PHONY: docs docs-serve docs-clean

# Auto-generate the API reference from NumPy-style docstrings in
# `thermal_model/`. Output is a static HTML site under `docs/api/`
# (gitignored). Open `docs/api/index.html` in a browser to read.
docs:
	pdoc thermal_model -o docs/api/ --docformat numpy --math

# Live-reload preview at http://localhost:8080 — useful while
# editing docstrings.
docs-serve:
	pdoc thermal_model --docformat numpy --math

docs-clean:
	rm -rf docs/api/
