NVCC = nvcc
CXXFLAGS = -O3 -arch=sm_75
TARGET = nbody
SRC = nbody.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CXXFLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET) *.out


