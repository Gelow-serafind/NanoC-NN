#ifndef NANOC_CASE_H
#define NANOC_CASE_H

#include <stdint.h>
#include "nanoc_status.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    NANOC_CASE_NONE = 0,
    NANOC_CASE_DUMMY_ECHO = 1,
    NANOC_CASE_MNIST_INT8 = 2,
    NANOC_CASE_TERMINAL_MODEL = 100,
} nanoc_case_id_t;

typedef nanoc_status_t (*nanoc_case_run_fn)(
    const uint8_t *input,
    uint32_t input_size,
    uint8_t *output,
    uint32_t output_capacity,
    uint32_t *output_size);

typedef struct {
    nanoc_case_id_t id;
    const char *name;
    uint32_t max_input_size;
    uint32_t max_output_size;
    nanoc_case_run_fn run;
} nanoc_case_desc_t;

const nanoc_case_desc_t *nanoc_case_find(nanoc_case_id_t id);
nanoc_status_t nanoc_case_run(
    nanoc_case_id_t id,
    const uint8_t *input,
    uint32_t input_size,
    uint8_t *output,
    uint32_t output_capacity,
    uint32_t *output_size,
    uint32_t *elapsed_us);

#ifdef __cplusplus
}
#endif

#endif /* NANOC_CASE_H */
