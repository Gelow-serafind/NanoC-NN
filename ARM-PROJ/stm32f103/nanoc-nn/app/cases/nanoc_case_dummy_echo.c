#include "nanoc_case_dummy_echo.h"

nanoc_status_t nanoc_case_dummy_echo_run(
    const uint8_t *input,
    uint32_t input_size,
    uint8_t *output,
    uint32_t output_capacity,
    uint32_t *output_size)
{
    uint32_t i;

    if (input_size > output_capacity) {
        return NANOC_STATUS_ERR_OUTPUT_SIZE;
    }

    for (i = 0U; i < input_size; ++i) {
        output[i] = input[i];
    }

    *output_size = input_size;
    return NANOC_STATUS_OK;
}
