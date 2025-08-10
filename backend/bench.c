#include <stddef.h>
#include <stdint.h>

size_t count_above_threshold(const uint8_t* data, size_t len);

static uint8_t bench_data[4096];

void setup_benchmark()
{
    for (int i = 0; i < 4096; i++)
    {
        bench_data[i] = (i * 37 + i/3) % 256;
    }
}

void run_benchmark()
{
    volatile size_t result = count_above_threshold(bench_data, 4096);
    (void)result;
}