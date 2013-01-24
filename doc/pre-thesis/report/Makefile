CMD=latexmk
FLAGS=-pvc

default:
	$(CMD) $(FLAGS)

%: tex/%.tex
	$(CMD) $(FLAGS) tex/$@.tex

clean:
	latexmk -C
