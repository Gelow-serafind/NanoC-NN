#ifndef NANOC_MODEL_H
#define NANOC_MODEL_H

#include <stdint.h>
#include <stddef.h>

#define NANOC_MODEL_INPUT_BYTES 64u
#define NANOC_MODEL_OUTPUT_BYTES 128u
#define NANOC_MODEL_INPUT_FLOATS 64u
#define NANOC_MODEL_OUTPUT_FLOATS 128u
#define NANOC_MODEL_ACTIVATION_A_BYTES 128u
#define NANOC_MODEL_ACTIVATION_B_BYTES 1u
#define NANOC_MODEL_SCRATCH_BYTES 400u
#define NANOC_MODEL_ESTIMATED_SRAM_BYTES 721u
#define NANOC_MODEL_ESTIMATED_FLASH_BYTES 3039u

typedef enum nanoc_status_t {
    NANOC_STATUS_OK = 0,
    NANOC_STATUS_BLOCKED = 2,
    NANOC_STATUS_UNSUPPORTED = 3,
    NANOC_STATUS_OVERSIZE = 4
} nanoc_status_t;

const char *nanoc_model_status(void);
int nanoc_model_run(const int8_t *input, int8_t *output);
int nanoc_model_run_float(const float * const *inputs, float *output);

#endif /* NANOC_MODEL_H */
