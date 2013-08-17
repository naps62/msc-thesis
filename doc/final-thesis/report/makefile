CMD=latexmk
FLAGS=-pvc -quiet

.PHONY: inputs default

default:
	$(CMD) $(FLAGS) main

%:
	find tex -regex "tex/$@.*\.tex" | xargs $(CMD) $(FLAGS)

clean:
	latexmk -C
	$(RM) *.bbl *.bib *.glo *.nls *.nlo *.tmp *.run.xml *.out *.aux *.blg *.fdb_latexmk *.log *.pdf
