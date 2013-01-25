CMD=latexmk
FLAGS=-pvc

default:
	$(CMD) $(FLAGS)

once:


%: tex/%.tex
	$(CMD) $(FLAGS) tex/$@.tex

clean:
	latexmk -C
