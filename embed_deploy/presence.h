#ifndef PRESENCE_H
#define PRESENCE_H

#include "ifx_sensor_dsp.h"
#include "math.h"

// #define XENSIV_BGT60TRXX_CONF_IMPL
// #include "radar_settings_embedded.h"

#define NUM_SAMPLES_PER_CHIRP (64)
#define NUM_CHIRPS_PER_FRAME  (16)
#define NUM_CHANNELS          (3)
#define NUM_RANGE_BINS        (NUM_SAMPLES_PER_CHIRP / 2)
#define NUM_DOPPLER_BINS      (NUM_CHIRPS_PER_FRAME)
#define FRAME_LEN             (NUM_SAMPLES_PER_CHIRP * NUM_CHIRPS_PER_FRAME * NUM_CHANNELS)
#define PREPROP_OUTPUT_SIZE   (NUM_RANGE_BINS * NUM_DOPPLER_BINS)

typedef struct predict_result_t {
    int class_index;
    float32_t probability;
} predict_result_t;

/* Function prototypes */

// intermediate_arrays init_intermediate_arrays(void);
// void free_intermediate_arrays(intermediate_arrays *inter_arrays);
// void fftshift_cf64(ifx_cf64_t *in, uint32_t len);
// void range_doppler_transform(float32_t *raw_frame, cfloat32_t *out, ifx_f32_t *range_window);
void presence_init(void);
void preprocess(uint16_t *samples, float32_t *processed_frames);
void predict(uint16_t *samples, float32_t *processed_frames, predict_result_t *results);
// void pplcount_run(float32_t *raw_frame, float32_t *processed_frames);
// predict_result_t predict(float32_t *input);

#endif