# Executable
EXE    = gcwe
	
# Compiler, Linker Defines
CC      = g++
CFLAGS  = -Wall -O3 -Wno-deprecated -m64 -I. -Wno-unused -std=c++11 
#LIBS    = ./srilm-lib/libsrilm.a ./srilm-lib/lm.a ./srilm-lib/misc.a ./srilm-lib/dstruct.a
#LIBS    = ./lib/lm.a ./lib/misc.a ./lib/dstruct.a
#LDFLAGS = $(LIBS)

# Compile and Assemble C++ Source Files into Object Files
%.o: %.cc
	echo compiling...$<
	$(CC) $(CFLAGS) -c $<
# -o $@
# Source and Object files
SRC    = $(wildcard *.cpp)
OBJ    = $(patsubst %.cpp, %.o, $(SRC))
OGCWE = GCWE.o

# Link all Object Files with external Libraries into Binaries
$(EXE): $(OBJ)
	echo linking...
	$(CC) $(CFLAGS) $(OGCWE) -o $(EXE) -lz

.PHONY: clean
clean:
	 -rm -f core *.o

