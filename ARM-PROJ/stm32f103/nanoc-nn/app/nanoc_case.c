#include "nanoc_case.h"

#include <stddef.h>

#include "cases/nanoc_case_dummy_echo.h"
#include "nanoc_config.h"
#include "nanoc_time.h"

static const nanoc_case_desc_t g_cases[] = {
    {
        NANOC_CASE_DUMMY_ECHO,
        "dummy_echo",
        NANOC_APP_MAX_INPUT_SIZE,
        NANOC_APP_MAX_OUTPUT_SIZE,
        nanoc_case_dummy_echo_run,
    },
};

const nanoc_case_desc_t *nanoc_case_find(nanoc_case_id_t id)
{
    uint32_t i;

    for (i = 0U; i < (uint32_t)(sizeof(g_cases) / sizeof(g_cases[0])); ++i) {
        if (g_cases[i].id == id) {
            return &g_cases[i];
        }
    }

    return NULL;
}

nanoc_status_t nanoc_case_run(
    nanoc_case_id_t id,
    const uint8_t *input,
    uint32_t input_size,
    uint8_t *output,
    uint32_t output_capacity,
    uint32_t *output_size,
    uint32_t *elapsed_us)
{
    const nanoc_case_desc_t *case_desc = nanoc_case_find(id);
    nanoc_time_us_t start_us;
    nanoc_time_us_t end_us;
    nanoc_status_t status;

    if (case_desc == NULL) {
        return NANOC_STATUS_ERR_UNKNOWN_CASE;
    }
    if (input_size > case_desc->max_input_size) {
        return NANOC_STATUS_ERR_INPUT_SIZE;
    }
    if (output_capacity < case_desc->max_output_size) {
        return NANOC_STATUS_ERR_OUTPUT_SIZE;
    }

    *output_size = 0U;
    *elapsed_us = 0U;

    start_us = nanoc_time_now_us();
    status = case_desc->run(input, input_size, output, output_capacity, output_size);
    end_us = nanoc_time_now_us();
    *elapsed_us = nanoc_time_elapsed_us32(start_us, end_us);

    return status;
}
