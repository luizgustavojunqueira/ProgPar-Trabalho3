CC = mpic++
CFLAGS = -Wall -g -Wno-int-to-pointer-cast 

.PHONY: clean main

main: clean main.cpp 
	${CC} ${CFLAGS} main.cpp -o main cliques.hpp cliques.cpp

clean:
	rm -f *.o *.a *.so
	rm -f main
	rm -f *.out
