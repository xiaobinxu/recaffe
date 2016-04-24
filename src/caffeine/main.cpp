#include "caffeine/syncedmem.hpp"
#include <assert.h>
#include <stdio.h>

using namespace caffeine;

int main()
{
  SyncedMemory mem(10);
  assert(mem.head() == SyncedMemory::UNINITIALIZED);
  assert(mem.cpu_data() != (void*)NULL);
  assert(mem.gpu_data() != (void*)NULL);

  printf("If you only see this line, then all tests are passed!\n");
  return 0;
} 
