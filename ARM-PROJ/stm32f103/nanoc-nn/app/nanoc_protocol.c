#include "nanoc_protocol.h"

#include "nanoc_uart.h"

#define NANOC_REQUEST_MAGIC 0x4454434EU
#define NANOC_RESPONSE_MAGIC 0x5254434EU
#define NANOC_REQUEST_HEADER_SIZE 16U
#define NANOC_RESPONSE_HEADER_SIZE 20U

static uint8_t g_header_buffer[NANOC_RESPONSE_HEADER_SIZE];

static uint16_t read_u16_le(const uint8_t *data)
{
    return (uint16_t)data[0] | ((uint16_t)data[1] << 8);
}

static uint32_t read_u32_le(const uint8_t *data)
{
    return (uint32_t)data[0]
        | ((uint32_t)data[1] << 8)
        | ((uint32_t)data[2] << 16)
        | ((uint32_t)data[3] << 24);
}

static void write_u16_le(uint8_t *data, uint16_t value)
{
    data[0] = (uint8_t)(value & 0xFFU);
    data[1] = (uint8_t)((value >> 8) & 0xFFU);
}

static void write_u32_le(uint8_t *data, uint32_t value)
{
    data[0] = (uint8_t)(value & 0xFFU);
    data[1] = (uint8_t)((value >> 8) & 0xFFU);
    data[2] = (uint8_t)((value >> 16) & 0xFFU);
    data[3] = (uint8_t)((value >> 24) & 0xFFU);
}

static uint32_t checksum_add(const uint8_t *data, uint32_t size)
{
    uint32_t i;
    uint32_t checksum = 0U;

    for (i = 0U; i < size; ++i) {
        checksum += data[i];
    }

    return checksum;
}

static uint32_t request_checksum(const uint8_t *header, const uint8_t *payload, uint32_t payload_size)
{
    return checksum_add(header, 12U) + checksum_add(payload, payload_size);
}

static uint32_t response_checksum(const uint8_t *header, const uint8_t *payload, uint32_t payload_size)
{
    return checksum_add(header, 16U) + checksum_add(payload, payload_size);
}

nanoc_status_t nanoc_protocol_read_request(nanoc_request_t *request, uint8_t *input_buffer)
{
    uint32_t magic;
    uint32_t expected_checksum;
    uint32_t actual_checksum;
    nanoc_status_t status;

    status = nanoc_uart_read_exact(g_header_buffer, NANOC_REQUEST_HEADER_SIZE);
    if (status != NANOC_STATUS_OK) {
        return status;
    }

    magic = read_u32_le(&g_header_buffer[0]);
    if (magic != NANOC_REQUEST_MAGIC || g_header_buffer[4] != NANOC_PROTOCOL_VERSION) {
        return NANOC_STATUS_ERR_PROTOCOL;
    }

    request->cmd = (nanoc_protocol_cmd_t)g_header_buffer[5];
    request->case_id = (nanoc_case_id_t)read_u16_le(&g_header_buffer[6]);
    request->input_size = read_u32_le(&g_header_buffer[8]);
    expected_checksum = read_u32_le(&g_header_buffer[12]);

    if (request->input_size > NANOC_PROTOCOL_MAX_INPUT_SIZE) {
        return NANOC_STATUS_ERR_INPUT_SIZE;
    }

    status = nanoc_uart_read_exact(input_buffer, request->input_size);
    if (status != NANOC_STATUS_OK) {
        return status;
    }

    actual_checksum = request_checksum(g_header_buffer, input_buffer, request->input_size);
    if (actual_checksum != expected_checksum) {
        return NANOC_STATUS_ERR_PROTOCOL;
    }

    return NANOC_STATUS_OK;
}

nanoc_status_t nanoc_protocol_write_response(const nanoc_response_t *response, const uint8_t *output_buffer)
{
    uint32_t output_size = response->output_size;
    uint32_t checksum;
    nanoc_status_t status;

    if (output_size > NANOC_PROTOCOL_MAX_OUTPUT_SIZE) {
        output_size = 0U;
    }

    write_u32_le(&g_header_buffer[0], NANOC_RESPONSE_MAGIC);
    g_header_buffer[4] = NANOC_PROTOCOL_VERSION;
    g_header_buffer[5] = (uint8_t)response->status;
    write_u16_le(&g_header_buffer[6], (uint16_t)response->case_id);
    write_u32_le(&g_header_buffer[8], response->elapsed_us);
    write_u32_le(&g_header_buffer[12], output_size);
    checksum = response_checksum(g_header_buffer, output_buffer, output_size);
    write_u32_le(&g_header_buffer[16], checksum);

    status = nanoc_uart_write_exact(g_header_buffer, NANOC_RESPONSE_HEADER_SIZE);
    if (status != NANOC_STATUS_OK) {
        return status;
    }

    if (output_size > 0U) {
        status = nanoc_uart_write_exact(output_buffer, output_size);
    }

    return status;
}

void nanoc_protocol_write_error(nanoc_status_t status)
{
    nanoc_response_t response;

    response.status = status;
    response.case_id = NANOC_CASE_NONE;
    response.elapsed_us = 0U;
    response.output_size = 0U;

    (void)nanoc_protocol_write_response(&response, 0);
}
