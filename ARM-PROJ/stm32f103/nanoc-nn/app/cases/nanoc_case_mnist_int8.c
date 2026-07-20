#include "nanoc_case_mnist_int8.h"

#include <stdint.h>

#define NANOC_MNIST_INPUT_BYTES 784U
#define NANOC_MNIST_OUTPUT_BYTES 10U

int nanoc_model_run(const int8_t *input, int8_t *output);

nanoc_status_t nanoc_case_mnist_int8_run(
    const uint8_t *input,
    uint32_t input_size,
    uint8_t *output,
    uint32_t output_capacity,
    uint32_t *output_size)
{
    int model_status;

    if (input_size != NANOC_MNIST_INPUT_BYTES) {
        return NANOC_STATUS_ERR_INPUT_SIZE;
    }
    if (output_capacity < NANOC_MNIST_OUTPUT_BYTES) {
        return NANOC_STATUS_ERR_OUTPUT_SIZE;
    }

    model_status = nanoc_model_run((const int8_t *)input, (int8_t *)output);
    if (model_status != 0) {
        return NANOC_STATUS_ERR_MODEL_RUN;
    }

    *output_size = NANOC_MNIST_OUTPUT_BYTES;
    return NANOC_STATUS_OK;
}
