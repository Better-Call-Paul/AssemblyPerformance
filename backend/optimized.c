#include <stddef.h>
#include <stdint.h>

size_t count_above_threshold(const uint8_t* data, size_t len)
{
    size_t count = 0;
    for (size_t i = 0; i < len; i+=8)
    {
        if (i+7<len)
        {
            count += data[i] > 128;
            count += data[i+1] > 128;
            count += data[i+2] > 128;
            count += data[i+3] > 128;
            count += data[i+4] > 128;
            count += data[i+5] > 128;
            count += data[i+6] > 128;
            count += data[i+7] > 128;
        }
        else
        {
            for (; i < len; ++i)
            {
                count += data[i] > 128;
            }
        }
    }
    return count;
}