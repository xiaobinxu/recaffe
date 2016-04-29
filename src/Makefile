PROJECT := caffe

NAME := lib$(PROJECT).so
TEST_NAME := test_$(PROJECT)

CXX_SRCS := $(shell find caffe ! -name "test_*.cpp" -name "*.cpp")
CU_SRCS := $(shell find caffe -name "*.cu")
TEST_SRCS := $(shell find caffe ! -name "test_caffe_main*.cpp" -name "test_*.cpp")
TEST_MAIN_SRCS := $(shell find caffe -name "test_caffe_main*.cpp")
GTEST_SRCS := gtest/gtest-all.cpp
PROGRAM_SRCS := $(shell find programs -name "*.cpp")

PROTO_SRCS := $(wildcard caffe/proto/*.proto)
PROTO_GEN_HEADER := $(PROTO_SRCS:.proto=.pb.h)
PROTO_GEN_CC := $(PROTO_SRCS:.proto=.pb.cc)
PROTO_GEN_PY := $(PROTO_SRCS:.proto=_pb2.py)

CXX_OBJS := $(CXX_SRCS:.cpp=.o)
CU_OBJS := $(CU_SRCS:.cu=.cuo)
PROTO_OBJS := $(PROTO_SRCS:.proto=.pb.o)
OBJS := $(PROTO_OBJS) $(CXX_OBJS) $(CU_OBJS)
TEST_OBJS := $(TEST_SRCS:.cpp=.o)
TEST_MAIN_OBJS := $(TEST_MAIN_SRCS:.cpp=.o)
GTEST_OBJS := $(GTEST_SRCS:.cpp=.o)
PROGRAM_OBJS := $(PROGRAM_SRCS:.cpp=.o)

TEST_BINS := $(TEST_OBJS:.o=.testbin)
PROGRAM_BINS := $(PROGRAM_OBJS:.o=.bin)

CUDA_DIR := /usr/local/cuda
CUDA_ARCH := -arch=sm_20
MKL_DIR := /opt/intel/mkl

CUDA_INCLUDE_DIR := $(CUDA_DIR)/include
CUDA_LIB_DIR := $(CUDA_DIR)/lib64
MKL_INCLUDE_DIR := $(MKL_DIR)/include
MKL_LIB_DIR := $(MKL_DIR)/lib/intel64

INCLUDE_DIRS := . /usr/local/include $(CUDA_INCLUDE_DIR) $(MKL_INCLUDE_DIR)
LIBRARY_DIRS := . /usr/local/lib $(CUDA_LIB_DIR) $(MKL_LIB_DIR)
LIBRARIES := cudart cublas protobuf glog mkl_rt mkl_intel_thread curand \
	           leveldb snappy opencv_core opencv_highgui pthread tcmalloc

WARNINGS += -Wall
CXXFLAGS += -fPIC
CPPFLAGS += $(foreach includedir,$(INCLUDE_DIRS),-I$(includedir))
LDFLAGS += $(foreach librarydir,$(LIBRARY_DIRS),-L$(librarydir))
LDFLAGS += $(foreach library,$(LIBRARIES),-l$(library))

LINK := $(CXX) $(CXXFLAGS) $(CPPFLAGS) $(WARNINGS)
NVCC := nvcc $(CXXFLAGS:-fPIC=-Xcompiler -fPIC) $(CPPFLAGS) $(CUDA_ARCH)

.PHONY: all test runtest program clean distclean linecount 

# TODO: temporarily build all test by default for convinience
#all: $(NAME)
all: $(TEST_NAME)

test: $(OBJS) $(GTEST_OBJS) $(TEST_MAIN_OBJS) $(TEST_BINS)

program: $(OBJS) $(PROGRAM_BINS)

runtest: test
	for testbin in $(TEST_BINS); do $$testbin 1; done

linecount: clean
	cloc --read-lang-def=caffe.cloc caffe/

$(NAME): $(PROTO_GEN_CC) $(OBJS)
	$(LINK) -shared $(OBJS) $(LDFLAGS) -o $(NAME)

$(CU_OBJS): %.cuo: %.cu
	$(NVCC) -c $< -o $@

$(PROTO_GEN_CC): $(PROTO_SRCS)
	protoc $(PROTO_SRCS) --cpp_out=. --python_out=.

$(TEST_NAME): $(NAME) $(TEST_OBJS) $(TEST_MAIN_OBJS)
	$(CXX) -o $(TEST_NAME) $(TEST_OBJS) $(TEST_MAIN_OBJS) $(GTEST_OBJS) $(LDFLAGS) $(WARNINGS) -l$(PROJECT) -pthread
	./$(TEST_NAME)

$(TEST_BINS): %.testbin : %.o
	$(CXX) -pthread $< $(OBJS) $(GTEST_OBJS) $(TEST_MAIN_OBJS) -o $@ $(LDFLAGS) $(WARNINGS)

$(PROGRAM_BINS): %.bin : %.o
	$(CXX) -pthread $< $(OBJS) -o $@ $(LDFLAGS) $(WARNINGS)

clean:
	@- $(RM) $(NAME) $(TEST_BINS) $(PROGRAM_BINS)
	@- $(RM) $(OBJS) $(TEST_OBJS) $(PROGRAM_OBJS)
	@- $(RM) $(PROTO_GEN_HEADER) $(PROTO_GEN_CC) $(PROTO_GEN_PY)
	@- $(RM) $(TEST_NAME) $(TEST_MAIN_OBJS)

distclean: clean

