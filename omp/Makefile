target: decide

CFLAGS= -g -O3 -std=c99 -Wall

aux.o: aux.c projet.h

decide: main.c aux.o
	$(CC) $(CFLAGS) -fopenmp $^ -o $@

exec: decide
	./decide "/ppp//PPP//7k//7K w"

.PHONY: clean
clean:
	rm -f *.o decide
