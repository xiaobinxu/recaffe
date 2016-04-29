PROJECT = caffe

NAME = lib$(PROJECT).so
TEST_NAME = test_$(PROJECT)

CXX_SRCS = $(shell find . ! -name "test_*.cpp" ! -name "gtest*" -name "*.cpp")
TEST_SRCS = $(shell find . -name "test_*.cpp") gtest/gtest-all.cpp
PROTO_SRCS = $(wildcard caffe/proto/*.proto)
PROTO_GEN_HEADER = $(PROTO_SRCS:.proto=.pb.h)
PROTO_GEN_CC = $(PROTO_SRCS:.proto=.pb.cc)
CXX_OBJS = $(CXX_SRCS:.cpp=.o)
PROTO_OBJS = $(PROTO_SRCS:.proto=.pb.o)
OBJS = $(PROTO_OBJS) $(CXX_OBJS)
TEST_OBJS = $(TEST_SRCS:.cpp=.o)

CUDA_DIR = /usr/local/cuda
MKL_DIR = /opt/intel/mkl

CUDA_INCLUDE_DIR = $(CUDA_DIR)/include
CUDA_LIB_DIR = $(CUDA_DIR)/lib64
MKL_INCLUDE_DIR = $(MKL_DIR)/include
MKL_LIB_DIR = $(MKL_DIR)/lib/intel64

INCLUDE_DIRS = . /usr/local/include $(CUDA_INCLUDE_DIR) $(MKL_INCLUDE_DIR)
LIBRARY_DIRS = . /usr/local/lib $(CUDA_LIB_DIR) $(MKL_LIB_DIR)
LIBRARIES = cudart cublas protobuf glog mkl_rt mkl_intel_thread

WARNINGS += -Wall
CXXFLAGS += -fPIC
CPPFLAGS += $(foreach includedir,$(INCLUDE_DIRS),-I$(includedir))
LDFLAGS += $(foreach librarydir,$(LIBRARY_DIRS),-L$(librarydir))
LDFLAGS += $(foreach library,$(LIBRARIES),-l$(library))

LINK = $(CXX) $(CXXFLAGS) $(CPPFLAGS) $(WARNINGS)

.PHONY: all test clean distclean linecount

all: $(NAME)

test: $(TEST_NAME)

linecount: clean
	cloc --read-lang-def=caffe.cloc caffe/

$(NAME): $(PROTO_GEN_CC) $(OBJS)
	$(LINK) -shared $(OBJS) $(LDFLAGS) -o $(NAME)

$(TEST_NAME): $(NAME) $(TEST_OBJS)
	$(CXX) -o $(TEST_NAME) $(TEST_OBJS) $(LDFLAGS) $(WARNINGS) -l$(PROJECT) -pthread
	./$(TEST_NAME)

$(PROTO_GEN_CC): $(PROTO_SRCS)
	protoc $(PROTO_SRCS) --cpp_out=.

clean:
	@- $(RM) $(NAME)
	@- $(RM) $(TEST_NAME)
	@- $(RM) $(OBJS)
	@- $(RM) $(TEST_OBJS)
	@- $(RM) $(PROTO_GEN_HEADER) $(PROTO_GEN_CC)

distclean: clean

