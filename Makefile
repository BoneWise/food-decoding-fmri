PYTHON ?= python
FONT_DIR ?= outputs/fonts
FIG_DIR ?= outputs/figures

.PHONY: tables validate stats fonts figures test reproduce

# Rebuild the manuscript tables from the reference bundle (no raw data needed).
tables:
	$(PYTHON) scripts/reproduce_results.py
# Check reference hashes and recompute archived quantities.
validate:
	$(PYTHON) scripts/validate_reference.py
# Paired k comparisons for the CNN/SVM decoders (reads results/reference only).
stats:
	$(PYTHON) analyses/paper_stats/s2_k_paired_tests.py
# TeX Gyre Heros as TrueType for matplotlib (needs TeX Live and fonttools).
fonts:
	mkdir -p $(FONT_DIR)
	for s in regular bold italic bolditalic; do \
	  $(PYTHON) analyses/figures/otf2ttf.py "$$(kpsewhich texgyreheros-$$s.otf)" $(FONT_DIR)/texgyreheros-$$s.ttf; done
# Figure 2 and Figure 4C-D from the reference bundle.
figures:
	$(PYTHON) analyses/figures/make_granularity_figure.py --summary results/reference/granularity --out $(FIG_DIR) --font-dir $(FONT_DIR)
	$(PYTHON) analyses/figures/make_fig4cd.py results/reference/figures results/reference/granularity $(FONT_DIR) $(FIG_DIR)
test:
	$(PYTHON) -m pytest -q
reproduce: tables validate stats figures
