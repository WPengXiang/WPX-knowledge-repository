
all: chapter-2-numpy.pdf


chapter-2-numpy.pdf: chapter-2-numpy.tex
	xelatex -shell-escape chapter-2-numpy.tex
	xelatex -shell-escape chapter-2-numpy.tex
	bibtex chapter-2-numpy.aux
	evince chapter-2-numpy.pdf&


.PHONY:clean  
clean:
	-rm -fr *.ps *.dvi *.aux *.toc *.idx *.ind *.ilg *.log *.out *~ *.tid *.tms *.pdf *.bak *.blg *.bbl *.gz *.snm *.nav _minte*

