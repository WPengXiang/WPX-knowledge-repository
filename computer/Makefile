all:install-ubuntu.pdf \
    study-vim.pdf \
	study-linux.pdf\
	study-git.pdf\
	install-matlab.pdf\
	install-mssoftware.pdf\
	assorted.pdf

install-mssoftware.pdf: install-mssoftware.tex
	xelatex -shell-escape install-mssoftware.tex
	xelatex -shell-escape install-mssoftware.tex
	bibtex install-mssoftware.aux
	evince install-mssoftware.pdf&

assorted.pdf: assorted.tex
	xelatex -shell-escape assorted.tex
	xelatex -shell-escape assorted.tex
	bibtex assorted.aux
	evince assorted.pdf&

install-ubuntu.pdf: install-ubuntu.tex
	xelatex -shell-escape install-ubuntu.tex
	xelatex -shell-escape install-ubuntu.tex
	bibtex install-ubuntu.aux
	evince install-ubuntu.pdf&


study-vim.pdf: study-vim.tex
	xelatex -shell-escape study-vim.tex
	xelatex -shell-escape study-vim.tex
	bibtex study-tex.aux
	evince study-vim.pdf&

install-matlab.pdf: install-matlab.tex
	xelatex -shell-escape install-matlab.tex
	xelatex -shell-escape install-matlab.tex
	bibtex install-matlab.aux
	evince install-matlab.pdf&

study-linux.pdf: study-linux.tex
	xelatex -shell-escape study-linux.tex
	xelatex -shell-escape study-linux.tex
	bibtex study-linux.aux
	evince study-linux.pdf&

study-git.pdf: study-git.tex
	xelatex -shell-escape study-git.tex
	xelatex -shell-escape study-git.tex
	bibtex study-git.aux
	evince study-git.pdf&

.PHONY:clean  
clean:
	-rm -fr *.ps *.dvi *.aux *.toc *.idx *.ind *.ilg *.log *.out *~ *.tid *.tms *.pdf *.bak *.blg *.bbl *.gz *.snm *.nav _minted*

