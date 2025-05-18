CXX = g++
CXXFLAGS = -std=c++17 -O3 -fopenmp
TARGET = test

all: $(TARGET)

$(TARGET): main.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

run: $(TARGET)
	./$(TARGET) /data/updated_flower.csv /data/updated_flower.csv 8

clean:
	rm -rf $(TARGET)

.PHONY: all clean run