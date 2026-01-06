CXX = g++
CXXFLAGS = -g -Wall -pthread -fopenmp

SRC = pthreads_omp.cpp \
      infrastructure/worker.cpp \
      infrastructure/thread_manager.cpp \
      ../helpers/image.cpp

INCLUDES = -I../helpers -Iinfrastructure

TARGET = pthreads_omp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) $(INCLUDES) -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
