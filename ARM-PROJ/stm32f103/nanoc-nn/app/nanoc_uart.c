#include "nanoc_uart.h"

#include "usart.h"

#define NANOC_UART_TIMEOUT_MS 10000U

void nanoc_uart_init(void)
{
}

nanoc_status_t nanoc_uart_read_exact(uint8_t *data, uint32_t size)
{
    if (HAL_UART_Receive(&huart1, data, (uint16_t)size, NANOC_UART_TIMEOUT_MS) != HAL_OK) {
        return NANOC_STATUS_ERR_UART;
    }

    return NANOC_STATUS_OK;
}

nanoc_status_t nanoc_uart_write_exact(const uint8_t *data, uint32_t size)
{
    if (HAL_UART_Transmit(&huart1, (uint8_t *)data, (uint16_t)size, NANOC_UART_TIMEOUT_MS) != HAL_OK) {
        return NANOC_STATUS_ERR_UART;
    }

    return NANOC_STATUS_OK;
}
