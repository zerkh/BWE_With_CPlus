#Executable
GCWE_EXE    = gcwe
EXE	= te
	
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

all:$(GCWE_EXE) $(OTE) $(EXE)
.PHONY : all

# Link all Object Files with external Libraries into Binaries
$(EXE): $(OBJ) 
		echo linking... 
		$(CC) $(CFLAGS) $(OBJ) $(LDFLAGS) -o $(EXE) -lz 

.PHONY: clean
clean:
	 -rm -f core *.o

