#include <stddef.h>
#include <stdint.h>

size_t count_above_threshold(const uint8_t* data, size_t len);

int run_tests()
{
    uint8_t test_data1[] = {100, 150, 200, 50, 130};
    size_t result = count_above_threshold(test_data1, 5);
    if (result != 3) return 1;
    
    uint8_t test_data2[] = {0, 1, 2, 3, 4, 5};
    result = count_above_threshold(test_data2, 6);
    if (result != 0) return 1;
    
    uint8_t test_data3[] = {255, 254, 253, 129, 128, 127};
    result = count_above_threshold(test_data3, 6);
    if (result != 4) return 1;
    
    uint8_t empty[] = {};
    result = count_above_threshold(empty, 0);
    if (result != 0) return 1;
    
    return 0;
}