CC = icc
FLAGS = -O3 -openmp -Wall -g
LIB = bksp
TARGET = lib$(LIB)
INCLUDE_DIR = include
LIBRARY_DIR = lib
SRC_DIR = src
BUILD_DIR = lib
CURRENT_DIR = $(shell pwd)
OBJ_DIR = objs
TEST_DIR = tests
SRCS := $(wildcard ./src/*.c)
OBJS := $(patsubst ./src%, ./objs%, $(SRCS:.c=.o))

nomkl =
OTHERLIBS =
OSX =
DLIBTYPE = .so
UNAME = $(shell uname -s)
LAPACK_DIR = /usr/local/lib

ifeq ($(UNAME), Darwin)
	OTHERLIBS = -framework Accelerate -L$(LAPACK_DIR) -llapacke
	DLIBTYPE = .dylib
	OSX = -DOSX
else
	OTHERLIBS = -lblas -llapacke
endif

ifeq ($(nomkl), yes)
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
	ld -shared -o ./$(BUILD_DIR)/$(TARGET)$(DLIBTYPE) $(OBJS) $(OTHERLIBS) -lc
	ar rcs ./$(BUILD_DIR)/$(TARGET).a $(OBJS)

./$(OBJ_DIR)/%.o : ./$(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c -fPIC -I./$(INCLUDE_DIR) -o $@ ./$(SRC_DIR)/$(@F:.o=.c)

test : ./$(TEST_DIR)/test.c
	$(CC) $(CFLAGS) -I./$(INCLUDE_DIR) -o ./$(TEST_DIR)/test ./$(TEST_DIR)/test.c -L$(CURRENT_DIR)/$(BUILD_DIR) -l$(LIB) $(OTHERLIBS) -lm

clean :
	rm -rf ./$(BUILD_DIR)/
	rm -rf ./$(OBJ_DIR)/

.PHONY : clean all

