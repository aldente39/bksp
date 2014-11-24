CC = icc
CFLAGS = -O3 -mkl -openmp -Wall -g
LIB = bksp
TARGET = lib$(LIB)
INCLUDE_DIR = include
LIBRARY_DIR = lib
SOURCE_DIR = src
BUILD_DIR = debug
TEST_DIR = test
SRCS = $(shell cd ./$(SOURCE_DIR) && ls *.c && cd ../)
OBJS = $(SRCS:.c=.o)

all : $(BUILD_DIR) $(patsubst %,$(BUILD_DIR)/%,$(TARGET))

$(BUILD_DIR) :
	mkdir $(BUILD_DIR)

$(BUILD_DIR)/$(TARGET) : $(patsubst %,$(SOURCE_DIR)/%,$(SRCS))
	$(CC) -shared -fPIC $(CFLAGS) -I./$(INCLUDE_DIR) -o $@.so $^

.PHONY : test
test :
	$(CC) $(CFLAGS) -p -I./$(INCLUDE_DIR) -o ./$(TEST_DIR)/test ./$(TEST_DIR)/test.c -L./$(BUILD_DIR) -lbksp

clean :
	rm ./$(BUILD_DIR)/*.so

