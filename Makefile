# Executable
GCWE_EXE    = gcwe
TE_EXE	= te
	
# Compiler, Linker Defines
CC      = g++
CFLAGS  = -Wall -O3 -Wno-deprecated -m64 -I. -Wno-unused -std=c++11 
#LIBS    = ./lib/lm.a ./lib/misc.a ./lib/dstruct.a
#LDFLAGS = $(LIBS)

# Compile and Assemble C++ Source Files into Object Files
%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

# Source and Object files
SRC    = $(wildcard *.cpp)
OBJ    = $(patsubst %.cpp, %.o, $(SRC))
OGCWE = GCWE.o
OTE = main.o

all:$(GCWE_EXE) $(TE_EXE)
.PHONY : all

# Link all Object Files with external Libraries into Binaries
$(GCWE_EXE): $(OGCWE)
	$(CC) $(CFLAGS) $(OGCWE) -o $(GCWE_EXE) -lz -lpthread
$(TE_EXE): $(OTE)
	$(CC) $(CFLAGS) $(OTE) -o $(TE_EXE) -lz -lpthread

.PHONY: clean
clean:
	 -rm -f core *.o

