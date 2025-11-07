# Compiler
CXX = mpic++
CXXFLAGS = -O2 -Wall -std=c++17 -I. -I./infrastructure -I../helpers -I../helpers/stb

# Output executable name
TARGET = mpi

# Source directories
SRC_DIRS = . ./infrastructure ../helpers ../helpers/stb

# Collect all .cpp files from the source directories
SRCS = $(foreach dir, $(SRC_DIRS), $(wildcard $(dir)/*.cpp))

# Generate a list of .o files corresponding to each source
OBJS = $(SRCS:.cpp=.o)

# Default rule
all: $(TARGET)

# Link all object files into the executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compile rule for .cpp -> .o
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Run rule
run: $(TARGET)
	mpirun -np $(NP) ./$(TARGET)

# Clean up build artifacts
clean:
	rm -f $(OBJS) $(TARGET)

# Rebuild everything
rebuild: clean all
