#ifndef NANOC_PROTOCOL_H
#define NANOC_PROTOCOL_H

#include <stdint.h>
#include "nanoc_case.h"
#include "nanoc_config.h"
#include "nanoc_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define NANOC_PROTOCOL_VERSION 1U
#define NANOC_PROTOCOL_MAX_INPUT_SIZE NANOC_APP_MAX_INPUT_SIZE
#define NANOC_PROTOCOL_MAX_OUTPUT_SIZE NANOC_APP_MAX_OUTPUT_SIZE

typedef enum {
    NANOC_CMD_RUN_CASE = 1,
} nanoc_protocol_cmd_t;

typedef struct {
    nanoc_protocol_cmd_t cmd;
    nanoc_case_id_t case_id;
    uint32_t input_size;
} nanoc_request_t;

typedef struct {
    nanoc_status_t status;
    nanoc_case_id_t case_id;
    uint32_t elapsed_us;
    uint32_t output_size;
} nanoc_response_t;

nanoc_status_t nanoc_protocol_read_request(nanoc_request_t *request, uint8_t *input_buffer);
nanoc_status_t nanoc_protocol_write_response(const nanoc_response_t *response, const uint8_t *output_buffer);
void nanoc_protocol_write_error(nanoc_status_t status);

#ifdef __cplusplus
}
#endif

#endif /* NANOC_PROTOCOL_H */
