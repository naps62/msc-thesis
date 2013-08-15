CMD=latexmk
FLAGS=-pvc -quiet

.PHONY: inputs default

default: inputs
	$(CMD) $(FLAGS) main

inputs:
	@echo "generating tex/_inputs.tex"
	@ls tex/ | egrep ^[^_] | sed "s/\([^.]*\)\.tex/\\\\subfile{tex\/\1}/" > tex/_inputs.tex

%:
	find tex -regex "tex/$@.*\.tex" | xargs $(CMD) $(FLAGS)

clean:
	latexmk -C
	$(RM) *.bbl *.bib *.glo *.nls *.nlo *.tmp *.run.xml *.out *.aux *.blg *.fdb_latexmk *.log *.pdf
