
%.pdf:  main.tex
	pdflatex $<
	- biber $*
	pdflatex $<
	pdflatex $<
	while ( grep -q '^LaTeX Warning: Label(s) may have changed' $*.log) \
	do pdflatex $<; done

main.pdf:

clean:
	latexmk -C
	rm -f comment.cut main.{aux,bbl,blg,log,out,pdf}
	rm -f *-converted-to.pdf

