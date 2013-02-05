CMD=latexmk
FLAGS=-pvc

default:
	$(CMD) $(FLAGS)

once:


%: tex/%.tex
	$(CMD) $(FLAGS) tex/$@.tex

clean:
	latexmk -C
	$(RM) *.bbl *.bib *.glo *.nls *.nlo *.tmp *.run.xml *.out
