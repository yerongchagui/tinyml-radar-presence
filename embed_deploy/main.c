/* Embedded Artificial Intelligence for Presence Detection using mmWave Radar */
/* Research Internship */

/* XENSIV BGT60TRXX guide: https://github.com/Infineon/sensor-xensiv-bgt60trxx */

#include <inttypes.h>
#include "cyhal.h"
#include "cybsp.h"
#include "cy_retarget_io.h"
#include "xensiv_bgt60trxx_mtb.h"
#define XENSIV_BGT60TRXX_CONF_IMPL
#include "radar_settings_embedded.h"
#include "presence.h"

/*******************************************************************************
* Macros
*******************************************************************************/
/* sensor SPI interface */
#define PIN_XENSIV_BGT60TRXX_SPI_SCLK       CYBSP_SPI_CLK
#define PIN_XENSIV_BGT60TRXX_SPI_MOSI       CYBSP_SPI_MOSI
#define PIN_XENSIV_BGT60TRXX_SPI_MISO       CYBSP_SPI_MISO
#define PIN_XENSIV_BGT60TRXX_SPI_CSN        CYBSP_SPI_CS

/* sensor interrupt output pin */
#define PIN_XENSIV_BGT60TRXX_IRQ            CYBSP_GPIO10
/* sensor HW reset pin */
#define PIN_XENSIV_BGT60TRXX_RSTN           CYBSP_GPIO11
/* enable 1V8 LDO on radar wingboard*/
#define PIN_XENSIV_BGT60TRXX_LDO_EN         CYBSP_GPIO5

#define XENSIV_BGT60TRXX_SPI_FREQUENCY      (25000000UL)

#define NUM_SAMPLES_PER_FRAME               (XENSIV_BGT60TRXX_CONF_NUM_RX_ANTENNAS *\
                                             XENSIV_BGT60TRXX_CONF_NUM_CHIRPS_PER_FRAME *\
                                             XENSIV_BGT60TRXX_CONF_NUM_SAMPLES_PER_CHIRP)

/*******************************************************************************
* Global variables
*******************************************************************************/
static cyhal_spi_t cyhal_spi;
static xensiv_bgt60trxx_mtb_t sensor;
static volatile bool data_available = false;

/* Allocate enough memory for the radar data frame. */
static uint16_t samples[NUM_SAMPLES_PER_FRAME];
// static float32_t presence_frame[NUM_SAMPLES_PER_FRAME];
float32_t processed_frames[PREPROP_OUTPUT_SIZE] = {0};

/* Interrupt handler to react on sensor indicating the availability of new data */
#if defined(CYHAL_API_VERSION) && (CYHAL_API_VERSION >= 2)
void xensiv_bgt60trxx_mtb_interrupt_handler(void *args, cyhal_gpio_event_t event)
#else
void xensiv_bgt60trxx_mtb_interrupt_handler(void *args, cyhal_gpio_irq_event_t event)
#endif
{
    CY_UNUSED_PARAMETER(args);
    CY_UNUSED_PARAMETER(event);
    data_available = true;
}

int main(void)
{
    cy_rslt_t result = CY_RSLT_SUCCESS;

    /* Initialize the device and board peripherals. */
    result = cybsp_init();
    CY_ASSERT(result == CY_RSLT_SUCCESS);

    __enable_irq();

    /* Initialize retarget-io to use the debug UART port. */
    result = cy_retarget_io_init(CYBSP_DEBUG_UART_TX, CYBSP_DEBUG_UART_RX, CY_RETARGET_IO_BAUDRATE);
    CY_ASSERT(result == CY_RSLT_SUCCESS);

    printf("XENSIV BGT60TRxx Example\r\n");

    /* Initialize the SPI interface to BGT60. */
    result = cyhal_spi_init(&cyhal_spi,
                            PIN_XENSIV_BGT60TRXX_SPI_MOSI,
                            PIN_XENSIV_BGT60TRXX_SPI_MISO,
                            PIN_XENSIV_BGT60TRXX_SPI_SCLK,
                            NC,
                            NULL,
                            8,
                            CYHAL_SPI_MODE_00_MSB,
                            false);
    CY_ASSERT(result == CY_RSLT_SUCCESS);

    /* Reduce drive strength to improve EMI */
    Cy_GPIO_SetSlewRate(CYHAL_GET_PORTADDR(PIN_XENSIV_BGT60TRXX_SPI_MOSI),
                        CYHAL_GET_PIN(PIN_XENSIV_BGT60TRXX_SPI_MOSI), CY_GPIO_SLEW_FAST);
    Cy_GPIO_SetDriveSel(CYHAL_GET_PORTADDR(PIN_XENSIV_BGT60TRXX_SPI_MOSI),
                        CYHAL_GET_PIN(PIN_XENSIV_BGT60TRXX_SPI_MOSI), CY_GPIO_DRIVE_1_8);
    Cy_GPIO_SetSlewRate(CYHAL_GET_PORTADDR(PIN_XENSIV_BGT60TRXX_SPI_SCLK),
                        CYHAL_GET_PIN(PIN_XENSIV_BGT60TRXX_SPI_SCLK), CY_GPIO_SLEW_FAST);
    Cy_GPIO_SetDriveSel(CYHAL_GET_PORTADDR(PIN_XENSIV_BGT60TRXX_SPI_SCLK),
                        CYHAL_GET_PIN(PIN_XENSIV_BGT60TRXX_SPI_SCLK), CY_GPIO_DRIVE_1_8);

    /* Set SPI data rate to communicate with sensor */
    result = cyhal_spi_set_frequency(&cyhal_spi, XENSIV_BGT60TRXX_SPI_FREQUENCY);
    CY_ASSERT(result == CY_RSLT_SUCCESS);

    /* Enable the LDO. */
    result = cyhal_gpio_init(PIN_XENSIV_BGT60TRXX_LDO_EN,
                             CYHAL_GPIO_DIR_OUTPUT,
                             CYHAL_GPIO_DRIVE_STRONG,
                             true);
    CY_ASSERT(result == CY_RSLT_SUCCESS);

    /* Wait LDO stable */
    (void)cyhal_system_delay_ms(5);

    result = xensiv_bgt60trxx_mtb_init(&sensor,
                                       &cyhal_spi,
                                       PIN_XENSIV_BGT60TRXX_SPI_CSN,
                                       PIN_XENSIV_BGT60TRXX_RSTN,
                                       register_list,
                                       XENSIV_BGT60TRXX_CONF_NUM_REGS);
    CY_ASSERT(result == CY_RSLT_SUCCESS);

    /* The sensor will generate an interrupt once the sensor FIFO level is
       NUM_SAMPLES_PER_FRAME */
    result = xensiv_bgt60trxx_mtb_interrupt_init(&sensor,
                                                 NUM_SAMPLES_PER_FRAME,
                                                 PIN_XENSIV_BGT60TRXX_IRQ,
                                                 CYHAL_ISR_PRIORITY_DEFAULT,
                                                 xensiv_bgt60trxx_mtb_interrupt_handler,
                                                 NULL);
    CY_ASSERT(result == CY_RSLT_SUCCESS);

    // /* Enable sensor data test mode. The data received on antenna RX1 will be overwritten by 
    //    a deterministic sequence of data generated by the test pattern generator */
    // if (xensiv_bgt60trxx_enable_data_test_mode(&sensor.dev, true) != XENSIV_BGT60TRXX_STATUS_OK)
    // {
    //     CY_ASSERT(0);
    // }

    if (xensiv_bgt60trxx_start_frame(&sensor.dev, true) != XENSIV_BGT60TRXX_STATUS_OK)
    {
        CY_ASSERT(0);
    }

    // pplcounting_init();
    presence_init();
    uint32_t frame_idx = 0;
    predict_result_t results;
    // inference_results_t results;
    // uint16_t test_word = XENSIV_BGT60TRXX_INITIAL_TEST_WORD;

    for(;;)
    {
        /* Wait for the radar device to indicate the availability of the data to fetch. */
        while (data_available == false);
        data_available = false;

        if (xensiv_bgt60trxx_get_fifo_data(&sensor.dev, samples,
                                           NUM_SAMPLES_PER_FRAME) == XENSIV_BGT60TRXX_STATUS_OK)
        {
            predict(samples, processed_frames, &results);

            // printf("%f, %f, %f, %f \n", presence_frame[0], presence_frame[1], (float32_t) samples[0] / 4095.0, (float32_t) samples[1] / 4095.0);
            // printf("Results: idx = %d, score = %f\n", results.idx, results.score);
            // CY_ASSERT(0);
            // /* Check received data */
            // for (int32_t sample_idx = 0; sample_idx < NUM_SAMPLES_PER_FRAME; ++sample_idx)
            // {
            //         if (test_word != samples[sample_idx])
            //         {
            //             printf("Frame %" PRIu32 " error detected. "
            //                    "Expected: %" PRIu16 ". "
            //                    "Received: %" PRIu16 "\n",
            //                    frame_idx, test_word, samples[sample_idx]);
            //             CY_ASSERT(0);
            //         }

            //         // Generate next test_word
            //         test_word = xensiv_bgt60trxx_get_next_test_word(test_word);
            // }
            printf("Frame %" PRIu32 " received correctly\n", frame_idx);
        }

        frame_idx++;
    }
}
