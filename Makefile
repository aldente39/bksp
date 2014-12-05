CC = icc
CFLAGS = -O3 -mkl -openmp -Wall -g
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

all : $(BUILD_DIR) $(OBJ_DIR) $(OBJS) $(BUILD_DIR)/$(TARGET)

$(OBJ_DIR) :
	mkdir -p $(OBJ_DIR)

$(BUILD_DIR) :
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/$(TARGET) : $(OBJS)
	ar rcs ./$(BUILD_DIR)/$(TARGET).a $(OBJS)

./$(OBJ_DIR)/%.o : ./$(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c -p -I./$(INCLUDE_DIR) -o $@ ./$(SRC_DIR)/$(@F:.o=.c)

test : ./$(TEST_DIR)/test.c
	$(CC) $(CFLAGS) -p -I./$(INCLUDE_DIR) -o ./$(TEST_DIR)/test ./$(TEST_DIR)/test.c -L./$(BUILD_DIR)/ -lbksp

clean :
	rm -rf ./$(BUILD_DIR)/
	rm -rf ./$(OBJ_DIR)/

.PHONY : clean all

