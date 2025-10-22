.PHONY: test paper

test:
	pytest -q

paper:
	cd docs/paper-latex && pdflatex -interaction=nonstopmode main.tex || true
