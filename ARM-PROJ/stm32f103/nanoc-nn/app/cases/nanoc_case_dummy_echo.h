#ifndef NANOC_CASE_DUMMY_ECHO_H
#define NANOC_CASE_DUMMY_ECHO_H

#include "nanoc_case.h"

#ifdef __cplusplus
extern "C" {
#endif

nanoc_status_t nanoc_case_dummy_echo_run(
    const uint8_t *input,
    uint32_t input_size,
    uint8_t *output,
    uint32_t output_capacity,
    uint32_t *output_size);

#ifdef __cplusplus
}
#endif

#endif /* NANOC_CASE_DUMMY_ECHO_H */
