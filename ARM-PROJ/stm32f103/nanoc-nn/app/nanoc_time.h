#ifndef NANOC_TIME_H
#define NANOC_TIME_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NANOC_TIME_TIM6_INPUT_HZ 72000000U
#define NANOC_TIME_TICK_HZ 1000000U
#define NANOC_TIME_TIM6_PRESCALER ((NANOC_TIME_TIM6_INPUT_HZ / NANOC_TIME_TICK_HZ) - 1U)

typedef uint64_t nanoc_time_us_t;

void nanoc_time_init(void);
nanoc_time_us_t nanoc_time_now_us(void);
uint32_t nanoc_time_now_us32(void);
uint32_t nanoc_time_elapsed_us32(nanoc_time_us_t start_us, nanoc_time_us_t end_us);
void nanoc_time_period_elapsed_callback(void *htim);

#ifdef __cplusplus
}
#endif

#endif /* NANOC_TIME_H */
