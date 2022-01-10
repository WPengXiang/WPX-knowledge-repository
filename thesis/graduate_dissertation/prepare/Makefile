all:1Wetting-phenomenon.pdf\
	2Model-generation.pdf\
	3Model-discretization.pdf

1Wetting-phenomenon.pdf: 1Wetting-phenomenon.tex
	xelatex -shell-escape 1Wetting-phenomenon.tex
	xelatex -shell-escape 1Wetting-phenomenon.tex
	bibtex 1Wetting-phenomenon.aux
	evince 1Wetting-phenomenon.pdf&

2Model-generation.pdf: 2Model-generation.tex
	xelatex -shell-escape 2Model-generation.tex
	xelatex -shell-escape 2Model-generation.tex
	bibtex 2Model-generation.aux
	evince 2Model-generation.pdf&

3Model-discretization.pdf: 3Model-discretization.tex
	xelatex -shell-escape 3Model-discretization.tex
	xelatex -shell-escape 3Model-discretization.tex
	bibtex 3Model-discretization.aux
	evince 3Model-discretization.pdf&

4Math-derivation.pdf: 4Math-derivation.tex
	xelatex -shell-escape 4Math-derivation.tex
	xelatex -shell-escape 4Math-derivation.tex
	bibtex 4Math-derivation.aux
	evince 4Math-derivation.pdf&

.PHONY:clean  
clean:
	-rm -fr *.ps *.dvi *.aux *.toc *.idx *.ind *.ilg *.log *.out *~ *.tid *.tms *.pdf *.bak *.blg *.bbl *.gz *.snm *.nav _minted*

