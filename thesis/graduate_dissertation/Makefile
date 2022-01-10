
all: graduate.pdf \

	 
graduate.pdf: graduate.tex
	xelatex -shell-escape graduate.tex
	bibtex graduate.aux
	xelatex -shell-escape graduate.tex
	xelatex -shell-escape graduate.tex
	evince graduate.pdf&

.PHONY:clean  
clean:
	-rm -fr *.ps *.dvi *.aux *.toc *.idx *.ind *.ilg *.log *.out *~ *.tid *.tms *.pdf *.bak *.blg *.bbl *.gz *.snm *.nav _minted*

