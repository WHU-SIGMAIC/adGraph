#include <hipcub/hipcub.hpp>

void b()
{
    printf("b() called\n");

    hipcub::DoubleBuffer<unsigned int>     d_keys;
    hipcub::DoubleBuffer<hipcub::NullType>    d_values;
    size_t                              temp_storage_bytes = 0;
    hipcub::DeviceRadixSort::SortPairs(NULL, temp_storage_bytes, d_keys, d_values, 1024);
}
