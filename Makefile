##### Please rewrite below parameters for your environment. #####

CC = icc
FLAGS = -O3 -openmp -Wall -g
### if you don't want to use Intel MKL, rewrite "mkl = yes" to "mkl = no". 
mkl = yes

BLAS_DIR = /usr/local/lib
LAPACK_DIR = /usr/local/lib

### In Mac OSX, if you don't want to use blas provided by Apple,
### rewrite "OSX_ACCE = yes" to "OSX_ACCE = no".
OSX_ACCE = yes


##### Please save below. #####

OBJ_DIR = objs
TEST_DIR = tests
SRCS := $(wildcard ./src/*.c)
OBJS := $(patsubst ./src%, ./objs%, $(SRCS:.c=.o))
LIB = bksp
TARGET = lib$(LIB)
INCLUDE_DIR = include
LIBRARY_DIR = lib
SRC_DIR = src
BUILD_DIR = lib
CURRENT_DIR = $(shell pwd)

OTHERLIBS =
OSX =
DLIBTYPE = .so
LD_ARG = -shared
UNAME = $(shell uname -s)


ifeq ($(UNAME), Darwin)
	ifeq ($(OSX_ACCE), yes)
		OTHERLIBS = -framework Accelerate -L$(LAPACK_DIR) -llapacke
	endif
	DLIBTYPE = .dylib
	OSX = -DOSX
	LD_ARG = -dylib
else
	OTHERLIBS =  -L$(BLAS_DIR) -lblas -L$(LAPACK_DIR) -llapacke
endif

ifeq ($(mkl), no)
	CFLAGS = $(FLAGS) -DNOMKL $(OSX)
else
	CFLAGS = $(FLAGS) -mkl $(OSX)
	OTHERLIBS = -L$(MKLROOT)/../compiler/lib/intel64  -L$(MKLROOT)/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -lpthread -liomp5
endif


all : library test

library : $(BUILD_DIR) $(OBJ_DIR) $(OBJS) $(BUILD_DIR)/$(TARGET)

$(OBJ_DIR) :
	mkdir -p $(OBJ_DIR)

$(BUILD_DIR) :
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/$(TARGET) : $(OBJS)
	ld $(LD_ARG) -o ./$(BUILD_DIR)/$(TARGET)$(DLIBTYPE) $(OBJS) $(OTHERLIBS) -lc
	ar rcs ./$(BUILD_DIR)/$(TARGET).a $(OBJS)

./$(OBJ_DIR)/%.o : ./$(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c -fPIC -I./$(INCLUDE_DIR) -o $@ ./$(SRC_DIR)/$(@F:.o=.c)

test : ./$(TEST_DIR)/test.c
	$(CC) $(CFLAGS) -I./$(INCLUDE_DIR) -o ./$(TEST_DIR)/test ./$(TEST_DIR)/test.c -L$(CURRENT_DIR)/$(BUILD_DIR) -l$(LIB) $(OTHERLIBS) -lm

clean :
	rm -rf ./$(BUILD_DIR)/
	rm -rf ./$(OBJ_DIR)/

.PHONY : clean all

