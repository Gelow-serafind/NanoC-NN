#ifndef NANOC_UART_H
#define NANOC_UART_H

#include <stdint.h>
#include "nanoc_status.h"

#ifdef __cplusplus
extern "C" {
#endif

void nanoc_uart_init(void);
nanoc_status_t nanoc_uart_read_exact(uint8_t *data, uint32_t size);
nanoc_status_t nanoc_uart_write_exact(const uint8_t *data, uint32_t size);

#ifdef __cplusplus
}
#endif

#endif /* NANOC_UART_H */
