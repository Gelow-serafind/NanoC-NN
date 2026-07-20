#include "model.h"
#include "model_weights.h"

#ifndef NANOC_ENABLE_CMSIS_NN
#define NANOC_ENABLE_CMSIS_NN 0
#endif

#if NANOC_ENABLE_CMSIS_NN
#include <math.h>
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"
#endif

static int8_t nanoc_activation_a[NANOC_MODEL_ACTIVATION_A_BYTES];
static int8_t nanoc_activation_b[NANOC_MODEL_ACTIVATION_B_BYTES];
static int8_t nanoc_scratch[NANOC_MODEL_SCRATCH_BYTES];
static int8_t nanoc_output_nhwc[128u];
static int8_t nanoc_tensor_conv_out[128u];

const char *nanoc_model_status(void)
{
    return "ok";
}

int nanoc_model_run(const int8_t *input, int8_t *output)
{
#if NANOC_ENABLE_CMSIS_NN
    arm_cmsis_nn_status cmsis_status;
    (void)nanoc_activation_a;
    (void)nanoc_activation_b;

    /* node 1: conv -> arm_convolve_wrapper_s8 */
    {
        cmsis_nn_context ctx;
        cmsis_nn_conv_params conv_params;
        cmsis_nn_per_channel_quant_params quant_params;
        cmsis_nn_dims input_dims = {1, 8, 8, 1};
        cmsis_nn_dims filter_dims = {8, 5, 5, 1};
        cmsis_nn_dims bias_dims = {1, 1, 1, 8};
        cmsis_nn_dims output_dims = {1, 4, 4, 8};

        conv_params.input_offset = 128;
        conv_params.output_offset = -128;
        conv_params.stride.h = 1;
        conv_params.stride.w = 1;
        conv_params.padding.h = 0;
        conv_params.padding.w = 0;
        conv_params.dilation.h = 1;
        conv_params.dilation.w = 1;
        conv_params.activation.min = -128;
        conv_params.activation.max = 127;
        quant_params.multiplier = (int32_t *)nanoc_conv_multiplier;
        quant_params.shift = (int32_t *)nanoc_conv_shift;

        ctx.buf = nanoc_scratch;
        ctx.size = arm_convolve_wrapper_s8_get_buffer_size(
            &conv_params, &input_dims, &filter_dims, &output_dims);
        if (ctx.size > NANOC_MODEL_SCRATCH_BYTES) {
            return NANOC_STATUS_BLOCKED;
        }

        cmsis_status = arm_convolve_wrapper_s8(
            &ctx,
            &conv_params,
            &quant_params,
            &input_dims,
            input,
            &filter_dims,
            nanoc_conv_weights,
            &bias_dims,
            nanoc_conv_bias,
            &output_dims,
            nanoc_tensor_conv_out);
        if (cmsis_status != ARM_CMSIS_NN_SUCCESS) {
            return NANOC_STATUS_BLOCKED;
        }
    }

    for (size_t nanoc_i = 0; nanoc_i < 128u; ++nanoc_i) {
        output[nanoc_i] = nanoc_tensor_conv_out[nanoc_i];
    }

    /* NHWC→NCHW output transpose (1x8x4x4) */
    for (int _ni = 0; _ni < 1; ++_ni)
        for (int _ci = 0; _ci < 8; ++_ci)
            for (int _hi = 0; _hi < 4; ++_hi)
                for (int _wi = 0; _wi < 4; ++_wi)
                    output[_ni * 128 + _ci * 16 + _hi * 4 + _wi] = nanoc_tensor_conv_out[_ni * 128 + _hi * 32 + _wi * 8 + _ci];
    return NANOC_STATUS_OK;
#else
    (void)input;
    (void)output;
    (void)nanoc_activation_a;
    (void)nanoc_activation_b;
    (void)nanoc_scratch;
    (void)nanoc_output_nhwc;
    (void)nanoc_tensor_conv_out;
    return NANOC_STATUS_BLOCKED;
#endif
}
