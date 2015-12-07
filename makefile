# Executable
#LEARN   = discourse_infering
#INFER	= discourse_decising
#CVSVM	= discourse_cvsvming
GCWE = train_gcwe

# Compiler, Linker Defines
CC      = g++
CFLAGS  = -Wall -O3 -Wno-deprecated -m64 -I. -Wno-unused -std=c++11
#CFLAGS  = -Wall -O -g -Wno-deprecated -m64 -I. -Wno-unused -std=c++11

# Compile and Assemble C++ Source Files into Object Files
%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@
# Source and Object files
SRC    = $(wildcard *.cpp)
OBJ    = $(patsubst %.cpp, %.o, $(SRC))
OLEARN = learning.o
OINFER = testing.o
OCVSVM = svming.o
	
# Link all Object Files with external Libraries into Binaries
all:$(INFER) $(LEARN) $(CVSVM)
.PHONY : all 

$(INFER): $(OINFER) 
	$(CC) $(CFLAGS) $(OINFER) liblbfgs.a -o $(INFER) -Wl,-Bstatic -lboost_regex -Wl,-Bdynamic -lz -lpthread 
$(CVSVM): $(OCVSVM) 
	$(CC) $(CFLAGS) $(OCVSVM) liblbfgs.a -o $(CVSVM) -Wl,-Bstatic -lboost_regex -Wl,-Bdynamic -lz -lpthread 
$(LEARN): $(OLEARN)
	$(CC) $(CFLAGS) $(OLEARN) liblbfgs.a -o $(LEARN)  -Wl,-Bstatic -lboost_regex -Wl,-Bdynamic -lz -lpthread 

.PHONY: clean
clean:
	 -rm -f core *.o

