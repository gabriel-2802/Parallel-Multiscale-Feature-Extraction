CXX = g++
CXXFLAGS = -Wall -std=c++17 -I. -I../helpers -I../helpers/stb -Wno-unused-but-set-variable

TARGET = serial

SRC_DIRS = . ../helpers ../helpers/stb
SRCS = $(foreach dir,$(SRC_DIRS),$(wildcard $(dir)/*.cpp))

vpath %.cpp $(SRC_DIRS)

OBJ_DIR = obj
OBJS = $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(notdir $(SRCS)))

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(OBJ_DIR)/%.o: %.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(OBJ_DIR)/*.o $(TARGET)

rebuild: clean all
