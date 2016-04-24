PROJECT = caffeine

NAME = $(PROJECT)_test

CXX_SRCS = $(wildcard caffeine/*.cpp)
PROTO_SRCS = $(wildcard caffeine/proto/*.proto)
PROTO_GEN_HEADER = $(PROTO_SRCS:.proto=.pb.h)
PROTO_GEN_CC = $(PROTO_SRCS:.proto=.pb.cc)
CXX_OBJS = $(CXX_SRCS:.cpp=.o)
PROTO_OBJS = $(PROTO_SRCS:.proto=.pb.o)
OBJS = $(CXX_OBJS) $(PROTO_OBJS)

CUDA_DIR = /usr/local/cuda
CUDA_INCLUDE_DIR = $(CUDA_DIR)/include
CUDA_LIB_DIR = $(CUDA_DIR)/lib64

INCLUDE_DIRS = . $(CUDA_INCLUDE_DIR)
LIBRARY_DIRS = . $(CUDA_LIB_DIR)
LIBRARIES = cudart cublas protobuf

WARNINGS += -Wall
#CXXFLAGS += -fPIC
CPPFLAGS += $(foreach includedir,$(INCLUDE_DIRS),-I$(includedir))
LDFLAGS += $(foreach librarydir,$(LIBRARY_DIRS),-L$(librarydir))
LDFLAGS += $(foreach library,$(LIBRARIES),-l$(library))

LINK = $(CXX) $(CXXFLAGS) $(CPPFLAGS) $(WARNINGS)

.PHONY: all clean distclean

all: $(NAME)

$(NAME): $(PROTO_GEN_CC) $(OBJS)
	$(LINK) $(OBJS) $(LDFLAGS) -o $(NAME)

$(PROTO_GEN_CC): $(PROTO_SRCS)
	protoc $(PROTO_SRCS) --cpp_out=.

clean:
	@- $(RM) $(NAME)
	@- $(RM) $(OBJS)
	@- $(RM) $(PROTO_GEN_HEADER) $(PROTO_GEN_CC)

distclean: clean

