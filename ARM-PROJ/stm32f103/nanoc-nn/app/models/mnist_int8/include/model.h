#ifndef NANOC_MODEL_H
#define NANOC_MODEL_H

#include <stdint.h>
#include <stddef.h>

#define NANOC_MODEL_INPUT_BYTES 784u
#define NANOC_MODEL_OUTPUT_BYTES 10u
#define NANOC_MODEL_INPUT_FLOATS 784u
#define NANOC_MODEL_OUTPUT_FLOATS 10u
#define NANOC_MODEL_ACTIVATION_A_BYTES 6272u
#define NANOC_MODEL_ACTIVATION_B_BYTES 3136u
#define NANOC_MODEL_SCRATCH_BYTES 6400u
#define NANOC_MODEL_ESTIMATED_SRAM_BYTES 16602u
#define NANOC_MODEL_ESTIMATED_FLASH_BYTES 37327u

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
