CC = nvcc

.PHONY: clean main

main: clean main.cu cliques.cu 
	$(CC) main.cu cliques.cu -o main
clean:
	rm -f *.o *.a *.so
	rm -f main
	rm -f *.out
