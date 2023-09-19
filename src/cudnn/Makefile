CXX := nvcc
CUDNN_PATH := /usr/local/cuda
SRC_DIR := src
OBJ_DIR := bin
HEADERS := -I $(CUDNN_PATH)/include -I ./include
LIBS := -L $(CUDNN_PATH)/lib64 -L/usr/local/lib
CXXFLAGS := -arch=sm_75 -std=c++11 -DDEBUG=0 -g
SRCS := Conv2d_Forward.cu Conv2d_Backward_Weight.cu Conv2d_Backward_Input.cu BatchNorm2d_Forward.cu BatchNorm2d_Backward.cu Dropout_Forward.cu Dropout_Backward.cu MaxPool2d_Forward.cu MaxPool2d_Backward.cu AdaptiveAvgPool2d_Forward.cu AdaptiveAvgPool2d_Backward.cu ReLU_Forward.cu ReLU_Backward.cu Scale_Forward.cu Scale_Backward.cu Linear_Forward.cu Linear_Backward_Bias.cu Linear_Backward_Weight.cu Linear_Backward_Input.cu Concat_Forward.cu Concat_Backward.cu Others.cu Utils.cu main.cu
OBJS=$(patsubst %.cu, $(OBJ_DIR)/%.o, $(SRCS))

$(shell mkdir -p bin)

all: main

main: $(OBJS)
	$(CXX) $(CXXFLAGS) $(HEADERS) $(LIBS) $(OBJS) -o main -lcudnn -lcurand -lcublas

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	${CXX} ${CXXFLAGS} $(HEADERS) $(LIBS) -c -o $@ $<

.phony: clean

clean:
	rm -f $(OBJ_DIR)/*.o main
