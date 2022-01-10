
all: chapter-1.pdf \
	 chapter-2.pdf \
	 chapter-3.pdf \
     chapter-4.pdf \
     chapter-5.pdf 

chapter-1.pdf: chapter-1.tex
	xelatex -shell-escape chapter-1.tex
	xelatex -shell-escape chapter-1.tex
	bibtex chapter-1.aux
	evince chapter-1.pdf&

chapter-2.pdf: chapter-2.tex
	xelatex -shell-escape chapter-2.tex
	xelatex -shell-escape chapter-2.tex
	bibtex chapter-2.aux
	evince chapter-2.pdf&

chapter-3.pdf: chapter-3.tex
	xelatex -shell-escape chapter-3.tex
	xelatex -shell-escape chapter-3.tex
	bibtex chapter-3.aux
	evince chapter-3.pdf&


chapter-4.pdf: chapter-4.tex
	xelatex -shell-escape chapter-4.tex
	xelatex -shell-escape chapter-4.tex
	bibtex chapter-4.aux
	evince chapter-4.pdf&


chapter-5.pdf: chapter-5.tex
	xelatex -shell-escape chapter-5.tex
	xelatex -shell-escape chapter-5.tex
	bibtex chapter-5.aux
	evince chapter-5.pdf&


.PHONY:clean  
clean:
	-rm -fr *.ps *.dvi *.aux *.toc *.idx *.ind *.ilg *.log *.out *~ *.tid *.tms *.pdf *.bak *.blg *.bbl *.gz *.snm *.nav _minte*

