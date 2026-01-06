CXX = g++
CXXFLAGS = -g -Wall -pthread

SRC = pthreads.cpp \
      infrastructure/worker.cpp \
      infrastructure/thread_manager.cpp \
      ../helpers/image.cpp

INCLUDES = -I../helpers -Iinfrastructure

TARGET = pthreads

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) $(INCLUDES) -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
