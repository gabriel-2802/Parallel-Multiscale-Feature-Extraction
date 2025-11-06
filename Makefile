# --- Compiler and Flags ---
CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -O2 -Wno-missing-field-initializers

# --- Directories ---
SRC_DIR := .
HELPER_DIR := helpers
OBJ_DIR := objects

# --- Files ---
SRC_FILES := $(wildcard $(SRC_DIR)/*.cpp) $(wildcard $(HELPER_DIR)/*.cpp)

# Replace .cpp with .o and move into ./objects/
OBJ_FILES := $(patsubst %.cpp, $(OBJ_DIR)/%.o, $(SRC_FILES))

# Output binary name
TARGET := main

# --- Default target ---
all: $(TARGET)

# --- Link ---
$(TARGET): $(OBJ_FILES)
	$(CXX) $(CXXFLAGS) -o $@ $^

# --- Compile each .cpp into ./objects/ folder ---
$(OBJ_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)       # ensure subfolders exist
	$(CXX) $(CXXFLAGS) -c $< -o $@

# --- Clean ---
clean:
	rm -rf $(OBJ_DIR) $(TARGET)

# --- Run ---
run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run
