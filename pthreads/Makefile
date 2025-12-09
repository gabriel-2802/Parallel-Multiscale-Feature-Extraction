CXX = g++
CXXFLAGS = -O2 -Wall -pthread

SRC = pthreads.cpp \
      infrastructure/worker.cpp \
      infrastructure/thread_manager.cpp \
      ../helpers/image.cpp 

INCLUDES = -Ihelpers -Iconvolution -Ipthreads

TARGET = pthreads

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) $(INCLUDES) -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
