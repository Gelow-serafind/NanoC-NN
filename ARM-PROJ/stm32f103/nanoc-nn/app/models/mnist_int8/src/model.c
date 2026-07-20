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
static int8_t nanoc_tensor_convolution28_quantizelinear_0[6272u];
static int8_t nanoc_tensor_pooling66_output_0_quantized[1568u];
static int8_t nanoc_tensor_convolution110_quantizelinear_0[3136u];
static int8_t nanoc_tensor_pooling160_output_0[256u];
static int8_t nanoc_tensor_plus214_output_0_matmul_quantizelinear_0[10u];
static int8_t nanoc_tensor_plus214_output_0_quantizelinear_0[10u];
static int8_t nanoc_tensor_pooling160_output_0_reshape0_gemm_matmul_quantizelinear[256u];
static int8_t nanoc_tensor_pooling160_output_0_reshape0[256u];

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

    /* node 1: Convolution28_quant -> arm_convolve_wrapper_s8 */
    {
        cmsis_nn_context ctx;
        cmsis_nn_conv_params conv_params;
        cmsis_nn_per_channel_quant_params quant_params;
        cmsis_nn_dims input_dims = {1, 28, 28, 1};
        cmsis_nn_dims filter_dims = {8, 5, 5, 1};
        cmsis_nn_dims bias_dims = {1, 1, 1, 8};
        cmsis_nn_dims output_dims = {1, 28, 28, 8};

        conv_params.input_offset = 128;
        conv_params.output_offset = -128;
        conv_params.stride.h = 1;
        conv_params.stride.w = 1;
        conv_params.padding.h = 2;
        conv_params.padding.w = 2;
        conv_params.dilation.h = 1;
        conv_params.dilation.w = 1;
        conv_params.activation.min = -128;
        conv_params.activation.max = 127;
        quant_params.multiplier = (int32_t *)nanoc_convolution28_quant_multiplier;
        quant_params.shift = (int32_t *)nanoc_convolution28_quant_shift;

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
            nanoc_convolution28_quant_weights,
            &bias_dims,
            nanoc_convolution28_quant_bias,
            &output_dims,
            nanoc_tensor_convolution28_quantizelinear_0);
        if (cmsis_status != ARM_CMSIS_NN_SUCCESS) {
            return NANOC_STATUS_BLOCKED;
        }
    }

    /* node 2: Pooling66_quant -> arm_max_pool_s8 */
    {
        cmsis_nn_context ctx;
        cmsis_nn_pool_params pool_params;
        cmsis_nn_dims input_dims = {1, 28, 28, 8};
        cmsis_nn_dims filter_dims = {1, 2, 2, 1};
        cmsis_nn_dims output_dims = {1, 14, 14, 8};

        pool_params.stride.h = 2;
        pool_params.stride.w = 2;
        pool_params.padding.h = 0;
        pool_params.padding.w = 0;
        pool_params.activation.min = -128;
        pool_params.activation.max = 127;
        ctx.buf = nanoc_scratch;
        ctx.size = 0;

        cmsis_status = arm_max_pool_s8(
            &ctx,
            &pool_params,
            &input_dims,
            nanoc_tensor_convolution28_quantizelinear_0,
            &filter_dims,
            &output_dims,
            nanoc_tensor_pooling66_output_0_quantized);
        if (cmsis_status != ARM_CMSIS_NN_SUCCESS) {
            return NANOC_STATUS_BLOCKED;
        }
    }

    /* node 3: Convolution110_quant -> arm_convolve_wrapper_s8 */
    {
        cmsis_nn_context ctx;
        cmsis_nn_conv_params conv_params;
        cmsis_nn_per_channel_quant_params quant_params;
        cmsis_nn_dims input_dims = {1, 14, 14, 8};
        cmsis_nn_dims filter_dims = {16, 5, 5, 8};
        cmsis_nn_dims bias_dims = {1, 1, 1, 16};
        cmsis_nn_dims output_dims = {1, 14, 14, 16};

        conv_params.input_offset = 128;
        conv_params.output_offset = -128;
        conv_params.stride.h = 1;
        conv_params.stride.w = 1;
        conv_params.padding.h = 2;
        conv_params.padding.w = 2;
        conv_params.dilation.h = 1;
        conv_params.dilation.w = 1;
        conv_params.activation.min = -128;
        conv_params.activation.max = 127;
        quant_params.multiplier = (int32_t *)nanoc_convolution110_quant_multiplier;
        quant_params.shift = (int32_t *)nanoc_convolution110_quant_shift;

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
            nanoc_tensor_pooling66_output_0_quantized,
            &filter_dims,
            nanoc_convolution110_quant_weights,
            &bias_dims,
            nanoc_convolution110_quant_bias,
            &output_dims,
            nanoc_tensor_convolution110_quantizelinear_0);
        if (cmsis_status != ARM_CMSIS_NN_SUCCESS) {
            return NANOC_STATUS_BLOCKED;
        }
    }

    /* node 5: Pooling160 -> arm_max_pool_s8 */
    {
        cmsis_nn_context ctx;
        cmsis_nn_pool_params pool_params;
        cmsis_nn_dims input_dims = {1, 14, 14, 16};
        cmsis_nn_dims filter_dims = {1, 3, 3, 1};
        cmsis_nn_dims output_dims = {1, 4, 4, 16};

        pool_params.stride.h = 3;
        pool_params.stride.w = 3;
        pool_params.padding.h = 0;
        pool_params.padding.w = 0;
        pool_params.activation.min = -128;
        pool_params.activation.max = 127;
        ctx.buf = nanoc_scratch;
        ctx.size = 0;

        cmsis_status = arm_max_pool_s8(
            &ctx,
            &pool_params,
            &input_dims,
            nanoc_tensor_convolution110_quantizelinear_0,
            &filter_dims,
            &output_dims,
            nanoc_tensor_pooling160_output_0);
        if (cmsis_status != ARM_CMSIS_NN_SUCCESS) {
            return NANOC_STATUS_BLOCKED;
        }
    }

    /* NCHW flatten for Pooling160_Output_0 -> nanoc_tensor_pooling160_output_0_reshape0_gemm_matmul_quantizelinear */
    for (int _ni = 0; _ni < 1; ++_ni)
        for (int _ci = 0; _ci < 16; ++_ci)
            for (int _hi = 0; _hi < 4; ++_hi)
                for (int _wi = 0; _wi < 4; ++_wi)
                    nanoc_tensor_pooling160_output_0_reshape0_gemm_matmul_quantizelinear[_ni * 256 + _ci * 16 + _hi * 4 + _wi] = nanoc_tensor_pooling160_output_0[_ni * 256 + _hi * 64 + _wi * 16 + _ci];

    /* node 8: gemm_MatMul_quant -> arm_fully_connected_per_channel_s8 */
    {
        cmsis_nn_context ctx;
        cmsis_nn_fc_params fc_params;
        cmsis_nn_per_channel_quant_params quant_params;
        cmsis_nn_dims input_dims = {1, 1, 1, 256};
        cmsis_nn_dims filter_dims = {256, 1, 1, 10};
        cmsis_nn_dims bias_dims = {1, 1, 1, 10};
        cmsis_nn_dims output_dims = {1, 1, 1, 10};

        ctx.buf = nanoc_scratch;
        ctx.size = arm_fully_connected_s8_get_buffer_size(&filter_dims);
        if (ctx.size > NANOC_MODEL_SCRATCH_BYTES) {
            return NANOC_STATUS_BLOCKED;
        }
        fc_params.input_offset = 128;
        fc_params.filter_offset = 0;
        fc_params.output_offset = -16;
        fc_params.activation.min = -128;
        fc_params.activation.max = 127;
        quant_params.multiplier = (int32_t *)nanoc_gemm_matmul_quant_multiplier;
        quant_params.shift = (int32_t *)nanoc_gemm_matmul_quant_shift;

        cmsis_status = arm_fully_connected_per_channel_s8(
            &ctx,
            &fc_params,
            &quant_params,
            &input_dims,
            nanoc_tensor_pooling160_output_0_reshape0_gemm_matmul_quantizelinear,
            &filter_dims,
            nanoc_gemm_matmul_quant_weights,
            &bias_dims,
            nanoc_gemm_matmul_quant_bias,
            &output_dims,
            nanoc_tensor_plus214_output_0_matmul_quantizelinear_0);
        if (cmsis_status != ARM_CMSIS_NN_SUCCESS) {
            return NANOC_STATUS_BLOCKED;
        }
    }

    /* node 9: gemm_Add_quant -> arm_elementwise_add_s8 */
    {
        cmsis_status = arm_elementwise_add_s8(
            nanoc_tensor_plus214_output_0_matmul_quantizelinear_0,
            nanoc_gemm_add_quant_weights,
            16,
            1073746813,
            1,
            7,
            1227711736,
            -15,
            0,
            nanoc_tensor_plus214_output_0_quantizelinear_0,
            -16,
            1073741824,
            1,
            -128,
            127,
            10);
        if (cmsis_status != ARM_CMSIS_NN_SUCCESS) {
            return NANOC_STATUS_BLOCKED;
        }
    }

    for (size_t nanoc_i = 0; nanoc_i < 10u; ++nanoc_i) {
        output[nanoc_i] = nanoc_tensor_plus214_output_0_quantizelinear_0[nanoc_i];
    }

    return NANOC_STATUS_OK;
#else
    (void)input;
    (void)output;
    (void)nanoc_activation_a;
    (void)nanoc_activation_b;
    (void)nanoc_scratch;
    (void)nanoc_tensor_convolution28_quantizelinear_0;
    (void)nanoc_tensor_pooling66_output_0_quantized;
    (void)nanoc_tensor_convolution110_quantizelinear_0;
    (void)nanoc_tensor_pooling160_output_0;
    (void)nanoc_tensor_plus214_output_0_matmul_quantizelinear_0;
    (void)nanoc_tensor_plus214_output_0_quantizelinear_0;
    (void)nanoc_tensor_pooling160_output_0_reshape0_gemm_matmul_quantizelinear;
    (void)nanoc_tensor_pooling160_output_0_reshape0;
    return NANOC_STATUS_BLOCKED;
#endif
}
