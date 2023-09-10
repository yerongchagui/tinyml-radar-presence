#include "wrap_tflite.h"
#include "presence.h"

static float32_t win_range[NUM_SAMPLES_PER_CHIRP];
// uint8_t buffer_counter = 0;
// bool buffer_filled = false;

void presence_init(void) {
  ifx_window_hamming_f32(win_range, NUM_SAMPLES_PER_CHIRP);
  tflite_init();
}

void preprocess(uint16_t *samples, float32_t *processed_frames) {
  float32_t raw_frames[FRAME_LEN];
  cfloat32_t range_array[NUM_RANGE_BINS * NUM_SAMPLES_PER_CHIRP];
  cfloat32_t doppler_array[NUM_DOPPLER_BINS * NUM_RANGE_BINS];

  for (int i = 0; i < FRAME_LEN; i++) {
    raw_frames[i] = (float32_t) samples[i] / 4095.0;
  }

  // Compute range-Doppler transform for each channel and find mean of values across channels
  for (int ch = 0; ch < NUM_CHANNELS; ch++) {
    ifx_range_fft_f32(raw_frames + ch * NUM_CHIRPS_PER_FRAME * NUM_SAMPLES_PER_CHIRP, 
                    range_array, true,
                    win_range, NUM_SAMPLES_PER_CHIRP,
                    NUM_CHIRPS_PER_FRAME);

    ifx_doppler_cfft_f32(range_array, doppler_array, false,
                          win_range, NUM_RANGE_BINS,
                          NUM_CHIRPS_PER_FRAME);
    
    ifx_shift_cfft_f32(doppler_array, NUM_DOPPLER_BINS, NUM_RANGE_BINS);

    // Re-use doppler_array to compute magnitude

    for (int i = 0; i < NUM_DOPPLER_BINS * NUM_RANGE_BINS; i++) {
      doppler_array[i] = sqrt(pow(creal(doppler_array[i]), 2) + pow(cimag(doppler_array[i]), 2));
    }

    for (int i = 0; i < NUM_DOPPLER_BINS * NUM_RANGE_BINS; i++) {
      processed_frames[i] += creal(doppler_array[i]) / NUM_CHANNELS;
    }
  }
}

void predict(uint16_t *samples, float32_t *processed_frames, predict_result_t *results) {
  preprocess(samples, processed_frames);
  run_inference(processed_frames, results);
}

// void preprocess(uint16_t *samples, float32_t *processed_frames) {
//   if (!init) {
//     ifx_window_hamming_f32(win_range, NUM_SAMPLES_PER_CHIRP);
//     init = true;
//   }

//   if (buffer_filled) {
//     for (int i = 0; i < FRAME_LEN; i++) {
//       raw_frames_buffer[i] = raw_frames_buffer[i + FRAME_LEN];
//     }
//   }

//   for (int i = buffer_counter * FRAME_LEN; i < FRAME_LEN; i++) {
//     raw_frames_buffer[i] = (float32_t) samples[i] / 4095.0;
//   }

//   if (buffer_counter < BUFFER_LEN - 1) buffer_counter++; 
//   else buffer_filled = true;

//   ifx_range_fft_f32(raw_frames_buffer, range_array, true,
//                       win_range, NUM_SAMPLES_PER_CHIRP,
//                       NUM_CHIRPS_PER_FRAME);

//   ifx_doppler_cfft_f32(range_array, doppler_array, false,
//                         win_range, NUM_RANGE_BINS,
//                         NUM_CHIRPS_PER_FRAME);
  
//   ifx_shift_cfft_f32(doppler_array, NUM_DOPPLER_BINS, NUM_RANGE_BINS);

//   // Re-use doppler_array (of type cfloat32_t) to compute magnitude

//   for (int i = 0; i < NUM_DOPPLER_BINS * NUM_RANGE_BINS; i++) {
//     doppler_array[i] = sqrt(pow(creal(doppler_array[i]), 2) + pow(cimag(doppler_array[i]), 2));
//   }

//   for (int i = 0; i < NUM_DOPPLER_BINS * NUM_RANGE_BINS; i++) {
//     processed_frames[i] = doppler_array[i];
//   }
// }

// // float32_t micro_frames[NUM_CHANNELS * NUM_MICRO_FRAMES * NUM_SAMPLES_PER_CHIRP] = {0};
// // float32_t marco_micro_frame[NUM_CHIRPS_PER_FRAME * MIN_RANGE_BINS * NUM_CHANNELS * 4] = {0};

// // intermediate_arrays init_intermediate_arrays(void) {
// //     uint32_t len_hfr = NUM_CHANNELS * NUM_CHIRPS_PER_FRAME * NUM_RANGE_BINS;
// //     uint32_t sz_f = sizeof(ifx_f32_t);
// //     uint32_t sz_c = sizeof(ifx_cf64_t);

// //     intermediate_arrays inter_arrays = {
// //         .x_range = (ifx_cf64_t *)malloc(sz_c * len_hfr),
// //         .x_doppler = (ifx_cf64_t *)malloc(sz_c * len_hfr),
// //         .doppler_window = (ifx_f32_t *)malloc(sz_f * NUM_CHIRPS_PER_FRAME),
// //         .range_window = (ifx_f32_t *)malloc(sz_f * NUM_SAMPLES_PER_CHIRP)
// //     };

// //     if (NUM_CHIRPS_PER_FRAME >= 16) 
// //         get_window(&WINDOWS.kaiser_b25, inter_arrays.doppler_window, NUM_CHIRPS_PER_FRAME);
// //     get_window(&WINDOWS.hann, inter_arrays.range_window, NUM_SAMPLES_PER_CHIRP);
// //     return inter_arrays;
// // }


// // void free_intermediate_arrays(intermediate_arrays *inter_arrays) {
// //     free(inter_arrays->x_range);
// //     free(inter_arrays->x_doppler);
// //     free(inter_arrays->doppler_window);
// //     free(inter_arrays->range_window);
// // }


// void fftshift_cf64(ifx_cf64_t *in, uint32_t len) {
//   assert(len % 2 == 0);
//   int half = len / 2;
//   ifx_cf64_t tmp;
//   for (int i = 0; i < half; ++i) {
//     // Swap elements at `i` and `i+half`
//     tmp = in[i];
//     in[i] = in[i + half];
//     in[i + half] = tmp;
//   }
// }

// void range_doppler_transform(float32_t *raw_frame, ifx_cf64_t *out, ifx_f32_t *range_window) {
//   uint16_t input_idx = 0;
//   uint16_t output_idx = 0;

//   uint16_t frame_size = NUM_CHANNELS * NUM_CHIRPS_PER_FRAME * NUM_SAMPLES_PER_CHIRP;
//   arm_scale_f32(
//     (float32_t *)raw_frame, 1.0 / (float32_t)ADC_NORMALIZATION,
//     (float32_t *)raw_frame, frame_size
//   );

//   for (int ch = 0; ch < NUM_CHANNELS; ++ch) {
//     uint32_t status = ifx_range_fft_f32((float32_t *)(raw_frame + input_idx), 
//                                         (cfloat32_t *)(out + output_idx), 
//                                          REMOVE_MEAN_TRUE, (float32_t *)range_window, 
//                                          NUM_SAMPLES_PER_CHIRP, NUM_CHIRPS_PER_FRAME);
//     if (status == IFX_SENSOR_DSP_ARGUMENT_ERROR) {
//       fprintf(stderr, "Range FFT: unsupported window size.\n");
//       abort();
//     }
//     fftshift_cf64(out + output_idx, NUM_CHIRPS_PER_FRAME * NUM_RANGE_BINS);

//     input_idx += NUM_CHIRPS_PER_FRAME * NUM_SAMPLES_PER_CHIRP;
//     output_idx += NUM_CHIRPS_PER_FRAME * NUM_RANGE_BINS;
//   }
// }

// void stack(float32_t *marco_micro_frame, ifx_cf64_t *macro_range, ifx_cf64_t *micro_doppler) {
//     for (int ch = 0; ch < NUM_CHANNELS; ch++) {
//         for (int chirp = 0; chirp < NUM_CHIRPS_PER_FRAME; chirp++) {
//             for (int range_bin_count = 0; range_bin_count < MIN_RANGE_BINS; range_bin_count++) {
//                 marco_micro_frame[chirp * MIN_RANGE_BINS * NUM_CHANNELS*4 + 
//                             range_bin_count * NUM_CHANNELS*4 + 
//                             ch] = macro_range[ch * NUM_CHIRPS_PER_FRAME * NUM_RANGE_BINS + 
//                                                             chirp * NUM_RANGE_BINS + 
//                                                             range_bin_count].data[0]; // Macro real value

//                 marco_micro_frame[chirp * MIN_RANGE_BINS * NUM_CHANNELS*4 + 
//                             range_bin_count * NUM_CHANNELS*4 +
//                             ch + 1] = macro_range[ch * NUM_CHIRPS_PER_FRAME * NUM_RANGE_BINS + 
//                                                             chirp * NUM_RANGE_BINS + 
//                                                             range_bin_count].data[1]; // Macro imaginary value

//                 marco_micro_frame[chirp * MIN_RANGE_BINS * NUM_CHANNELS*4 + 
//                             range_bin_count * NUM_CHANNELS*4 +
//                             ch + 2] = micro_doppler[ch * NUM_CHIRPS_PER_FRAME * NUM_RANGE_BINS +
//                                                             chirp * NUM_RANGE_BINS +
//                                                             range_bin_count].data[0]; // Micro real value

//                 marco_micro_frame[chirp * MIN_RANGE_BINS * NUM_CHANNELS*4 + 
//                             range_bin_count * NUM_CHANNELS*4 +
//                             ch + 3] = micro_doppler[ch * NUM_CHIRPS_PER_FRAME * NUM_RANGE_BINS +
//                                                             chirp * NUM_RANGE_BINS +
//                                                             range_bin_count].data[1]; // Micro imaginary value
//             }
//         }
//     }
// }


// // void preprocess(uint16_t *samples, float32_t *processed_frames) {

// //     /*
    
// //     float32_t* raw_frame[3072]      [0, 1, ..., 2047] are the micro_frames, and [2048, 2049, ... 3071] is
// //                                     the range FFT of the raw_frame
                                  
    
// //     */

// //     /* Find mean along chirps */

// //     float32_t raw_frame[NUM_SAMPLES_PER_CHIRP * NUM_CHANNELS * NUM_CHIRPS_PER_FRAME];

// //     for (int i = 0; i < NUM_SAMPLES_PER_CHIRP * NUM_CHANNELS * NUM_CHIRPS_PER_FRAME; i++) {
// //         raw_frame[i] = (float32_t) samples[i] / 4095.0;
// //     }

// //     float32_t mean_along_chirps[NUM_CHANNELS * NUM_SAMPLES_PER_CHIRP];

// //     for (int ch = 0; ch < NUM_CHANNELS; ch++) {
// //         for (int s = 0; s < NUM_SAMPLES_PER_CHIRP; s++) {
// //             float32_t sum = 0;
// //             for (int chirp = 0; chirp < NUM_CHIRPS_PER_FRAME; chirp++) {
// //                 sum += raw_frame[ch * NUM_SAMPLES_PER_CHIRP * NUM_CHIRPS_PER_FRAME + 
// //                                chirp * NUM_SAMPLES_PER_CHIRP + s] / NUM_CHIRPS_PER_FRAME;
// //             }
// //             mean_along_chirps[ch * NUM_SAMPLES_PER_CHIRP + s] = sum;
// //         }
// //     }

// //     /* Apply the range-doppler transform on the current raw frame (Macro) */

// //     ifx_cf64_t *x_range = (ifx_cf64_t *)malloc(sizeof(ifx_cf64_t) * NUM_CHANNELS * NUM_CHIRPS_PER_FRAME * NUM_RANGE_BINS);
// //     // ifx_cf64_t x_range[NUM_CHANNELS * NUM_CHIRPS_PER_FRAME * NUM_RANGE_BINS];
// //     ifx_f32_t *range_window = (ifx_f32_t *)malloc(sizeof(ifx_f32_t) * NUM_SAMPLES_PER_CHIRP);
// //     // ifx_f32_t range_window[NUM_SAMPLES_PER_CHIRP];
// //     get_window(&WINDOWS.hann, range_window, NUM_SAMPLES_PER_CHIRP);
// //     range_doppler_transform(raw_frame, x_range, range_window);
// //     // for (int i = 0; i < sizeof(ifx_cf64_t) * NUM_CHANNELS * NUM_CHIRPS_PER_FRAME * NUM_RANGE_BINS; i++) {
// //     //     raw_frame[i] = x_range[i / 2].data[i % 2];
// //     // }

// //     /* Compute the Doppler FFT (Micro) */

// //     ifx_cf64_t *x_doppler = (ifx_cf64_t *)malloc(sizeof(ifx_cf64_t) * NUM_CHANNELS * NUM_CHIRPS_PER_FRAME * NUM_RANGE_BINS);
// //     // ifx_cf64_t x_doppler[NUM_CHANNELS * NUM_CHIRPS_PER_FRAME * NUM_RANGE_BINS];
// //     float32_t micro_frames[NUM_CHANNELS * NUM_MICRO_FRAMES * NUM_SAMPLES_PER_CHIRP] = {0};

// //     for (int ch = 0; ch < NUM_CHANNELS; ch++) {
// //         for (int s = 0; s < NUM_SAMPLES_PER_CHIRP; s++) {
// //             // Shift the 2nd-16th frames to 1st-15th
// //             int f = 0;
// //             for ( ; f < NUM_MICRO_FRAMES - 1; f++) {
// //                 micro_frames[ch * NUM_SAMPLES_PER_CHIRP * NUM_MICRO_FRAMES + 
// //                        f * NUM_SAMPLES_PER_CHIRP + s] = 
// //                 micro_frames[ch * NUM_SAMPLES_PER_CHIRP * NUM_MICRO_FRAMES + 
// //                        (f + 1) * NUM_SAMPLES_PER_CHIRP + s];
// //             }
// //             // Add the new mean frame to 16th frame
// //             micro_frames[ch * NUM_SAMPLES_PER_CHIRP * NUM_MICRO_FRAMES + 
// //                        f * NUM_SAMPLES_PER_CHIRP + s] = mean_along_chirps[ch * NUM_SAMPLES_PER_CHIRP + s];
// //         }
// //     }

// //     /* Apply the range-doppler transform on the micro frames */

// //     range_doppler_transform(micro_frames, x_doppler, range_window);
// //     free(range_window);

// //     /* Stack macro and micro frames */

// //     // float32_t macro_micro_frame[NUM_CHIRPS_PER_FRAME * MIN_RANGE_BINS * NUM_CHANNELS * 4] = {0};

// //     // stack(macro_micro_frame, x_range, x_doppler);
// //     free(x_range);
// //     free(x_doppler);

// //     for (int chirp = 0; chirp < NUM_CHIRPS_PER_FRAME; chirp++) {
// //         for (int range_bin_count = 0; range_bin_count < MIN_RANGE_BINS; range_bin_count++) {
// //             for (int ch = 0; ch < NUM_CHANNELS * 4; ch++) {
// //                 int f = 0;
// //                 for (; f < NUM_STACKED_FRAMES - 1; f++) {
// //                     // Shift the 2nd-8th frames to 1st-7th
// //                     processed_frames[f * NUM_CHIRPS_PER_FRAME * MIN_RANGE_BINS * NUM_CHANNELS*4 +
// //                                 chirp * MIN_RANGE_BINS * NUM_CHANNELS*4 +
// //                                 range_bin_count * NUM_CHANNELS*4 +
// //                                 ch] =
// //                                 processed_frames[(f + 1) * NUM_CHIRPS_PER_FRAME * MIN_RANGE_BINS * NUM_CHANNELS*4 +
// //                                         chirp * MIN_RANGE_BINS * NUM_CHANNELS*4 +
// //                                         range_bin_count * NUM_CHANNELS*4 +
// //                                         ch];
// //                 }
// //                 // Append the new stacked frame to the 8th
// //                 processed_frames[f * NUM_CHIRPS_PER_FRAME * MIN_RANGE_BINS * NUM_CHANNELS*4 +
// //                             chirp * MIN_RANGE_BINS * NUM_CHANNELS*4 +
// //                             range_bin_count * NUM_CHANNELS*4 +
// //                             ch] =
// //                             micro_frames[chirp * MIN_RANGE_BINS * NUM_CHANNELS*4 +
// //                                                 range_bin_count * NUM_CHANNELS*4 +
// //                                                 ch];
// //             }
// //         }
// //     }
// // }

// // void pplcount_preprocess_old(float32_t *raw_frame, intermediate_arrays *inter_arrays, float32_t *processed_frames) {

// //     /*
    
// //     float32_t* raw_frame[3072]      [0, 1, ..., 2047] are the micro_frames, and [2048, 2049, ... 3071] is
// //                                     the range FFT of the raw_frame
                                  
    
// //     */

// //     /* Find mean along chirps */
// //     // Can make use of dynamic memory allocation

// //     float32_t mean_along_chirps[NUM_CHANNELS * NUM_SAMPLES_PER_CHIRP];
// //     for (int ch = 0; ch < NUM_CHANNELS; ch++) {
// //         for (int s = 0; s < NUM_SAMPLES_PER_CHIRP; s++) {
// //             float32_t sum = 0;
// //             for (int chirp = 0; chirp < NUM_CHIRPS_PER_FRAME; chirp++) {
// //                 sum += raw_frame[ch * NUM_SAMPLES_PER_CHIRP * NUM_CHIRPS_PER_FRAME + 
// //                                chirp * NUM_SAMPLES_PER_CHIRP + s] / NUM_CHIRPS_PER_FRAME;
// //             }
// //             mean_along_chirps[ch * NUM_SAMPLES_PER_CHIRP + s] = sum;
// //         }
// //     }

// //     /* Create micro frames */

// //     for (int ch = 0; ch < NUM_CHANNELS; ch++) {
// //         for (int s = 0; s < NUM_SAMPLES_PER_CHIRP; s++) {
// //             // Shift the 2nd-16th frames to 1st-15th
// //             int f = 0;
// //             for ( ; f < NUM_MICRO_FRAMES - 1; f++) {
// //                 micro_frames[ch * NUM_SAMPLES_PER_CHIRP * NUM_MICRO_FRAMES + 
// //                        f * NUM_SAMPLES_PER_CHIRP + s] = 
// //                 micro_frames[ch * NUM_SAMPLES_PER_CHIRP * NUM_MICRO_FRAMES + 
// //                        (f + 1) * NUM_SAMPLES_PER_CHIRP + s];
// //             }
// //             // Add the new mean frame to 16th frame
// //             micro_frames[ch * NUM_SAMPLES_PER_CHIRP * NUM_MICRO_FRAMES + 
// //                        f * NUM_SAMPLES_PER_CHIRP + s] = mean_along_chirps[ch * NUM_SAMPLES_PER_CHIRP + s];
// //         }
// //     }

// //     /* Apply the range-doppler transform on the current raw frame (Macro) */

// //     range_doppler_transform(raw_frame, inter_arrays->x_range, inter_arrays->range_window);

// //     /* Apply the range-doppler transform on the micro frames */

// //     range_doppler_transform(micro_frames, inter_arrays->x_doppler, inter_arrays->range_window);

// //     /* Stack macro and micro frames */

// //     stack(marco_micro_frame, inter_arrays->x_range, inter_arrays->x_doppler);

// //     for (int chirp = 0; chirp < NUM_CHIRPS_PER_FRAME; chirp++) {
// //         for (int range_bin_count = 0; range_bin_count < MIN_RANGE_BINS; range_bin_count++) {
// //             for (int ch = 0; ch < NUM_CHANNELS * 4; ch++) {
// //                 int f = 0;
// //                 for (; f < NUM_STACKED_FRAMES - 1; f++) {
// //                     // Shift the 2nd-8th frames to 1st-7th
// //                     processed_frames[f * NUM_CHIRPS_PER_FRAME * MIN_RANGE_BINS * NUM_CHANNELS*4 +
// //                                 chirp * MIN_RANGE_BINS * NUM_CHANNELS*4 +
// //                                 range_bin_count * NUM_CHANNELS*4 +
// //                                 ch] =
// //                                 processed_frames[(f + 1) * NUM_CHIRPS_PER_FRAME * MIN_RANGE_BINS * NUM_CHANNELS*4 +
// //                                         chirp * MIN_RANGE_BINS * NUM_CHANNELS*4 +
// //                                         range_bin_count * NUM_CHANNELS*4 +
// //                                         ch];
// //                 }
// //                 // Append the new stacked frame to the 8th
// //                 processed_frames[f * NUM_CHIRPS_PER_FRAME * MIN_RANGE_BINS * NUM_CHANNELS*4 +
// //                             chirp * MIN_RANGE_BINS * NUM_CHANNELS*4 +
// //                             range_bin_count * NUM_CHANNELS*4 +
// //                             ch] =
// //                             marco_micro_frame[chirp * MIN_RANGE_BINS * NUM_CHANNELS*4 +
// //                                                 range_bin_count * NUM_CHANNELS*4 +
// //                                                 ch];
// //             }
// //         }
// //     }
// // }

// // void pplcount_run(float32_t *raw_frame, float32_t *processed_frames) {
// //     // intermediate_arrays inter_arrays;
// //     // inter_arrays = init_intermediate_arrays();
// //     pplcount_preprocess(raw_frame, processed_frames);
// // }