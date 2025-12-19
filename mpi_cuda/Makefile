CXX = mpic++
NVCC = nvcc

MPI_INC = /usr/lib/x86_64-linux-gnu/openmpi/include

CXXFLAGS = -Wall -std=c++17 -I. -I./infrastructure -I../helpers -I../helpers/stb
NVCCFLAGS = -I. -I./infrastructure -I../helpers -I../helpers/stb -I$(MPI_INC) -std=c++17

CUDA_ARCH = -arch=sm_86

TARGET = mpi_cuda

SRC_DIRS = . ./infrastructure ../helpers ../helpers/stb
CPP_SRCS = $(foreach dir,$(SRC_DIRS),$(wildcard $(dir)/*.cpp))

vpath %.cpp $(SRC_DIRS)

OBJ_DIR = obj
CPP_OBJS = $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(notdir $(CPP_SRCS)))

all: $(TARGET)

$(TARGET): $(CPP_OBJS)
	$(NVCC) $(CUDA_ARCH) -o $@ $^ -ccbin mpic++ -lcudart

$(OBJ_DIR)/%.o: %.cpp | $(OBJ_DIR)
	$(NVCC) $(CUDA_ARCH) $(NVCCFLAGS) -x cu -dc $< -o $@

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

run: $(TARGET)
	mpirun --oversubscribe -np 8 ./$(TARGET)

run_custom: $(TARGET)
	mpirun --oversubscribe -np $(NP) ./$(TARGET)

clean:
	rm -f $(OBJ_DIR)/*.o $(TARGET)

rebuild: clean all
