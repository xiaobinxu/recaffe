#ifndef CAFFINE_COMMON_HPP_
#define CAFFINE_COMMON_HPP_

#include <cstddef>
#include <iostream>
#include <boost/shared_ptr.hpp>
#include <glog/logging.h>
#include "driver_types.h"

namespace caffeine {
    using boost::shared_ptr;
    using std::size_t;
}

static std::ostream nullout(0);

#define CUDA_CHECK(condition) \
  CHECK((condition) == cudaSuccess)

#endif

