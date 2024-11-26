CC = g++
CFLAGS = -Wall -g -Wno-int-to-pointer-cast 

.PHONY: clean main

main: clean main.cpp 
	$(CC) $(CFLAGS) -o main main.cpp 
clean:
	rm -f *.o *.a *.so
	rm -f main
	rm -f *.out
