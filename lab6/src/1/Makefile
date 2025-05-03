# Makefile for parallel_for library and main (matrix multiplication)

CC        := gcc
CFLAGS    := -fPIC -O2 -pthread
LDFLAGS   := -shared
LIB       := libparallel_for.so
EXE       := main

.PHONY: all clean

all: $(LIB) $(EXE)

# 构建动态库
$(LIB): parallel_for.o
	$(CC) $(LDFLAGS) -o $@ $^

parallel_for.o: parallel_for.c parallel_for.h
	$(CC) $(CFLAGS) -c -o $@ parallel_for.c

# 构建主程序
$(EXE): main.o $(LIB)
	$(CC) -O2 -pthread -o $@ main.o -L. -lparallel_for

main.o: main.c parallel_for.h
	$(CC) $(CFLAGS) -c -o $@ main.c

clean:
	@rm -f *.o $(LIB) $(EXE) bench_results.txt
