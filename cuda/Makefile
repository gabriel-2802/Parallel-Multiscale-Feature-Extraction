CXX = nvcc
CXXFLAGS = -arch=sm_86 -I. -I./infrastructure -I../helpers -I../helpers/stb

TARGET = cuda

CUDA_MAIN = cuda.cu
IMAGE_SRC = ../helpers/image.cpp

all: $(TARGET)

$(TARGET): $(CUDA_MAIN) $(IMAGE_SRC)
	$(CXX) $(CXXFLAGS) -o $@ $(CUDA_MAIN) $(IMAGE_SRC)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

rebuild: clean all
