TARGETS=medianfilter meanfilter

all: $(TARGETS)

medianfilter: medianfilter.c
	gccx -o medianfilter medianfilter.c

meanfilter: meanfilter.c
	gccx -o meanfilter meanfilter.c

clean:
	rm -f *.0 $(TARGETS) *~
