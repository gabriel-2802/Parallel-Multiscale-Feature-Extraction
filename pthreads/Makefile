CXX = g++
CXXFLAGS = -O2 -Wall -pthread

SRC = pthreads.cpp \
      worker/worker.cpp \
      convolution/convolution.cpp \
      helpers/image.cpp 

INCLUDES = -Ihelpers -Iconvolution -Ipthreads

TARGET = app

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) $(INCLUDES) -o $(TARGET)

clean:
	rm -f $(TARGET)
