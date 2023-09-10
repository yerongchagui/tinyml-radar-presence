/* Expose a C friendly interface for main functions */ 
/* This is crucial. Otherwise, you will run into library import errors */

#ifndef WRAP_TFLITE_H_
#define WRAP_TFLITE_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "presence.h"

void tflite_init(void);
void run_inference(float32_t *input, predict_result_t *results);

#ifdef __cplusplus
}
#endif
#endif