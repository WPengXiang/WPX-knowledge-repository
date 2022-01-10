
all: weekly-report.pdf \

weekly-report.pdf: weekly-report.tex
	xelatex -shell-escape weekly-report.tex
	xelatex -shell-escape weekly-report.tex
	evince weekly-report.pdf&

.PHONY:clean  
clean:
	-rm -fr  */*.bcf */*.mtc */*.maf */*.mtc* */*.xml */*.ps */*.dvi */*.aux */*.toc */*.idx */*.ind */*.ilg */*.log */*.out */*~ */*.tid */*.tms */*.pdf */*.bak */*.blg */*.bbl */*.gz */*.snm */*.nav */_minted*

