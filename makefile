TARGETS=median

all: $(TARGETS)

median: medianfilter.c
	gccx -o medianfilter medianfilter.c

clean:
	rm -f *.0 $(TARGETS) *~
