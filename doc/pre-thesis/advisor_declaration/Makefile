CMD=latexmk
FLAGS=-pvc -quiet

default:
	$(CMD) $(FLAGS)

%: tex/%.tex
	$(CMD) $(FLAGS) tex/$@.tex

clean:
	latexmk -C
	$(RM) *.bbl *.bib *.glo *.nls *.nlo *.tmp *.run.xml *.out
