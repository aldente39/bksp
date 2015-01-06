CC = icc
FLAGS = -O3 -openmp -Wall -g
LIB = bksp
TARGET = lib$(LIB)
INCLUDE_DIR = include
LIBRARY_DIR = lib
SRC_DIR = src
BUILD_DIR = lib
OBJ_DIR = objs
TEST_DIR = tests
SRCS := $(wildcard ./src/*.c)
OBJS := $(patsubst ./src%, ./objs%, $(SRCS:.c=.o))

nomkl =
OTHERLIBS =
UNAME = $(shell uname -s)
LAPACK_DIR = /usr/local/lib

ifeq ($(UNAME), Darwin)
	OTHERLIBS = -DOSX -framework Accelerate -L$(LAPACK_DIR) -llapacke
else
	OTHERLIBS = -lblas -llapacke
endif

ifeq ($(nomkl), yes)
	CFLAGS = $(FLAGS) -DNOMKL
else
	CFLAGS = $(FLAGS) -mkl
	OTHERLIBS = 
endif


all : library test

library : $(BUILD_DIR) $(OBJ_DIR) $(OBJS) $(BUILD_DIR)/$(TARGET)

$(OBJ_DIR) :
	mkdir -p $(OBJ_DIR)

$(BUILD_DIR) :
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/$(TARGET) : $(OBJS)
	ar rcs ./$(BUILD_DIR)/$(TARGET).a $(OBJS)

./$(OBJ_DIR)/%.o : ./$(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c -I./$(INCLUDE_DIR) -o $@ ./$(SRC_DIR)/$(@F:.o=.c) $(OTHERLIBS) -lm

test : ./$(TEST_DIR)/test.c
	$(CC) $(CFLAGS) -I./$(INCLUDE_DIR) -o ./$(TEST_DIR)/test ./$(TEST_DIR)/test.c -L./$(BUILD_DIR) -l$(LIB) $(OTHERLIBS) -lm

clean :
	rm -rf ./$(BUILD_DIR)/
	rm -rf ./$(OBJ_DIR)/

.PHONY : clean all

