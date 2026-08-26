CXX = g++

CXXFLAGS = -std=c++17 -Wall -Wextra -Wpedantic -Iheaders -O2

TARGET = bin/main.exe

SRC = main.cpp \
      Neuron.cpp \
      Layer.cpp \
      Network.cpp \
      MNIST.cpp \
	  test.cpp

OBJ = $(SRC:.cpp=.o)

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(OBJ)
	@mkdir -p bin .out
	$(CXX) $(CXXFLAGS) $(OBJ) -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(OBJ)
	rm -f $(TARGET)
	rm -f .out/image.txt
	rm -f .out/*.png