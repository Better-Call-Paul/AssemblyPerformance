#include <stddef.h>
#include <stdint.h>

size_t count_above_threshold(const uint8_t* data, size_t len)
{
    size_t count = 0;
    for (size_t i = 0; i < len; i++)
    {
        if (data[i] > 128)
        {
            count++;
        }
    }
    return count;
}