#include "nanoc_app.h"

#include "stm32f1xx_hal.h"

#include "nanoc_case.h"
#include "nanoc_config.h"
#include "nanoc_protocol.h"
#include "nanoc_time.h"
#include "nanoc_uart.h"

static uint8_t g_input_buffer[NANOC_PROTOCOL_MAX_INPUT_SIZE];
static uint8_t g_output_buffer[NANOC_PROTOCOL_MAX_OUTPUT_SIZE];

typedef enum {
    NANOC_APP_STATE_UART_HEARTBEAT = 0,
    NANOC_APP_STATE_PROTOCOL_RUNNER = 1,
} nanoc_app_state_t;

static nanoc_app_state_t g_app_state = NANOC_APP_STATE_PROTOCOL_RUNNER;

static uint32_t nanoc_app_append_text(uint8_t *buffer, uint32_t offset, const char *text)
{
    while (*text != '\0') {
        buffer[offset++] = (uint8_t)*text++;
    }

    return offset;
}

static uint32_t nanoc_app_append_u32(uint8_t *buffer, uint32_t offset, uint32_t value)
{
    char digits[10];
    uint32_t count = 0U;

    if (value == 0U) {
        buffer[offset++] = (uint8_t)'0';
        return offset;
    }

    while (value > 0U) {
        digits[count++] = (char)('0' + (value % 10U));
        value /= 10U;
    }

    while (count > 0U) {
        buffer[offset++] = (uint8_t)digits[--count];
    }

    return offset;
}

void nanoc_app_init(void)
{
    nanoc_uart_init();
    nanoc_time_init();

#if NANOC_APP_UART_HEARTBEAT_TEST
    g_app_state = NANOC_APP_STATE_UART_HEARTBEAT;
#else
    g_app_state = NANOC_APP_STATE_PROTOCOL_RUNNER;
#endif
}

void nanoc_app_poll(void)
{
#if NANOC_APP_UART_HEARTBEAT_TEST
    uint8_t heartbeat[64];
    uint32_t heartbeat_size = 0U;

    if (g_app_state == NANOC_APP_STATE_UART_HEARTBEAT) {
        heartbeat_size = nanoc_app_append_text(heartbeat, heartbeat_size, "NanoC-NN F103 UART heartbeat t=");
        heartbeat_size = nanoc_app_append_u32(heartbeat, heartbeat_size, nanoc_time_now_us32());
        heartbeat_size = nanoc_app_append_text(heartbeat, heartbeat_size, " us\r\n");
        (void)nanoc_uart_write_exact(heartbeat, heartbeat_size);
        HAL_Delay(1000U);
        return;
    }
#endif

    nanoc_request_t request;
    nanoc_response_t response;
    uint32_t output_size = 0U;
    uint32_t elapsed_us = 0U;

    nanoc_status_t status = nanoc_protocol_read_request(&request, g_input_buffer);
    if (status != NANOC_STATUS_OK) {
        nanoc_protocol_write_error(status);
        return;
    }

    if (request.cmd != NANOC_CMD_RUN_CASE) {
        nanoc_protocol_write_error(NANOC_STATUS_ERR_UNKNOWN_CMD);
        return;
    }

    status = nanoc_case_run(
        request.case_id,
        g_input_buffer,
        request.input_size,
        g_output_buffer,
        sizeof(g_output_buffer),
        &output_size,
        &elapsed_us);

    response.status = status;
    response.case_id = request.case_id;
    response.elapsed_us = elapsed_us;
    response.output_size = (status == NANOC_STATUS_OK) ? output_size : 0U;

    (void)nanoc_protocol_write_response(&response, g_output_buffer);
}
