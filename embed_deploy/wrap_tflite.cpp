#define TF_LITE_STATIC_MEMORY
#define TF_LITE_MCU_DEBUG_LOG
#include "arm_math_types.h"
#include "wrap_tflite.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "models/c_model.h"

#define ARENA_SIZE (10 * 1024)
uint8_t tensor_arena[ARENA_SIZE];

tflite::MicroInterpreter *p_interpreter;
static tflite::MicroErrorReporter micro_error_reporter;
const tflite::Model *model;
static tflite::AllOpsResolver resolver;

TfLiteTensor *model_input;
TfLiteTensor *model_output;

void tflite_init(void) {
  // Load model
  model = tflite::GetModel(c_model_tflite);

  p_interpreter = new tflite::MicroInterpreter(model, resolver, tensor_arena, ARENA_SIZE, &micro_error_reporter);
  TfLiteStatus allocate_status = p_interpreter->AllocateTensors();

  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(&micro_error_reporter, "AllocateTensors() failed");
    while(1);
  }

  model_input = p_interpreter->input(0);
  model_output = p_interpreter->output(0);

  printf("Testing!");
}

void run_inference(float32_t *input, predict_result_t *results) {
  model_input->data.f = input;
  TfLiteStatus invoke_status = p_interpreter->Invoke();

  if (invoke_status != kTfLiteOk) TF_LITE_REPORT_ERROR(&micro_error_reporter, "Invoke failed\n");

  results->class_index = model_output->data.f[0];
  // this code is kept here so we can use it in case we want to switch to quantized NNs
  // results->score = (pred->data.int8[results->idx]-pred->params.zero_point)*pred->params.scale;
  results->probability = 0.0; //pred->data.f[results->idx];
}