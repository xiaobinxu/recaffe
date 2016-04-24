#ifndef CAFFINE_COMMON_HPP_
#define CAFFINE_COMMON_HPP_

#include <cstddef>
#include <iostream>
#include <boost/shared_ptr.hpp>
#include "driver_types.h"

namespace caffeine {
    using boost::shared_ptr;
    using std::size_t;
}

static std::ostream nullout(0);

#define LOG_IF(condition) \
    ((condition) == cudaSuccess) ? nullout : std::cout

#define CUDA_CHECK(condition) \
    LOG_IF(condition) << "Check failed: " #condition " "

#endif

