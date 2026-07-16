#include "nanoc_time.h"

#include "tim.h"

static volatile uint32_t g_tim6_overflows;

void nanoc_time_init(void)
{
    g_tim6_overflows = 0U;
    __HAL_TIM_SET_COUNTER(&htim6, 0U);
    (void)HAL_TIM_Base_Start_IT(&htim6);
}

nanoc_time_us_t nanoc_time_now_us(void)
{
    uint32_t primask;
    uint32_t overflows;
    uint32_t counter;

    primask = __get_PRIMASK();
    __disable_irq();

    overflows = g_tim6_overflows;
    counter = __HAL_TIM_GET_COUNTER(&htim6);

    if ((__HAL_TIM_GET_FLAG(&htim6, TIM_FLAG_UPDATE) != RESET) && counter < 32768U) {
        ++overflows;
    }

    if (primask == 0U) {
        __enable_irq();
    }

    return (((nanoc_time_us_t)overflows) << 16) + (nanoc_time_us_t)counter;
}

uint32_t nanoc_time_now_us32(void)
{
    return (uint32_t)nanoc_time_now_us();
}

uint32_t nanoc_time_elapsed_us32(nanoc_time_us_t start_us, nanoc_time_us_t end_us)
{
    return (uint32_t)(end_us - start_us);
}

void nanoc_time_period_elapsed_callback(void *htim)
{
    if (htim == &htim6) {
        ++g_tim6_overflows;
    }
}

void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim)
{
    nanoc_time_period_elapsed_callback(htim);
}
