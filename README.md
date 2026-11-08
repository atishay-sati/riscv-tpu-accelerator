# RISC-V TPU Accelerator

![Neural Network Accelerator Architecture](images/Neural_Network_Accelerator_Architecture_Infographic.png)

An INT8 neural network accelerator implemented in SystemVerilog on a **Digilent Basys 3 FPGA**. The project combines a custom 8×8 systolic array, dense-layer controllers, banked on-chip memory, and UART communication to perform neural network inference on hardware.

The goal is to explore how specialized hardware accelerates neural network workloads compared with running the same model in software.

## Project Overview

The accelerator runs a quantized neural network using an 8-bit integer datapath.

The current network architecture consists of:

- **Input:** 784 values (28×28 image)
- **Hidden layer:** 128 neurons
- **Output layer:** 32 classes
- **Arithmetic:** INT8 weights and activations with wider accumulators
- **Compute engine:** Shared 8×8 systolic array
- **Processing elements:** 64

A Python application allows a user to draw an image, run local inference using TensorFlow Lite, and send the same image to the FPGA for hardware inference.

The FPGA returns its predicted class over UART.

## Hardware Architecture

The main hardware components are:

| Component | Purpose |
|---|---|
| `tpu_top.sv` | Top-level neural network accelerator |
| `systolic_array_8x8.sv` | 8×8 parallel multiply-accumulate array |
| `systolic_pe.sv` | Individual systolic processing element |
| `dense_layer_controller.sv` | Controls dense-layer computation and weight loading |
| `dense_tile_controller.sv` | Coordinates tiled matrix operations |
| `requantize_int8.sv` | Converts accumulated values back to INT8 |
| `uart_rx.sv` | Receives serial data |
| `uart_tx.sv` | Transmits serial data |
| `uart_tpu_controller.sv` | Coordinates image reception and inference |
| `uart_tpu_system.sv` | Integrates UART and TPU hardware |

### Tiled Computation

The accelerator uses an 8×8 systolic array to perform matrix multiplication.

Because the neural network dimensions exceed the array dimensions, computations are divided into smaller tiles.

The same systolic array is reused for the hidden and output layers, reducing the amount of hardware required.

### Weight Banking and Parallel Memory Access

To improve systolic array loading efficiency, the accelerator uses an **eight-bank weight memory architecture**.

Instead of storing all 64 weights for an 8-neuron tile in a single memory structure that must be accessed sequentially, the weights are distributed across eight independent memory banks.

Each bank supplies weights for one processing element column of the systolic array.

For a neuron index \(n\) and input feature index \(f\), the memory mapping is:

```text
Bank address = n % 8
Memory address = floor(n / 8) * F + f
```

Where:

- `n` is the output neuron index.
- `f` is the input feature index.
- `F` is the number of input features in the layer.
- Each bank stores weights for neurons assigned to that bank.

This organization allows all eight banks to be accessed simultaneously for a given input feature.

As a result, the controller can retrieve the weights needed for eight neurons in parallel rather than reading each neuron's weights individually.

#### Weight Loading Optimization

The `dense_layer_controller.sv` manages the banked memory access sequence.

The optimized loading process uses three stages:

1. `LOAD_W_ADDR`: Generates the addresses for the eight weight banks.
2. `LOAD_W_WAIT`: Allows the synchronous memory reads to complete.
3. `LOAD_W_SAVE`: Captures the returned weights into the appropriate registers.

For an eight-neuron tile, the weight-loading process requires eight feature-column reads, with three controller cycles per feature column.

| Loading operation | Original approach | Banked approach |
|---|---:|---:|
| Weight memory reads | 64 | 8 |
| Cycles per read | 3 | 3 |
| Total weight-loading cycles | 192 | 24 |
| Activation-loading cycles | 24 | 24 |
| Total tile-loading cycles | 216 | 48 |

This reduces the total tile-loading overhead from 216 cycles to 48 cycles, a **4.5× reduction in loading cycles**.

The optimization improves memory parallelism without increasing the number of processing elements. It also allows the same banked memory architecture to support both the hidden and output layers.

### Numeric Representation

The network uses INT8 quantized weights and activations. Accumulation uses wider arithmetic to preserve intermediate results before requantization.

Weights, biases, multipliers, and shift values are initialized using `.mem` files.

These memory initialization files are incorporated into the FPGA design during synthesis.

## Computer-to-FPGA Communication

The computer communicates with the Basys 3 through USB-UART.

The communication protocol is:

**Computer → FPGA**

1. Send packet header: `0xAA`.
2. Send 784 image bytes, representing a 28×28 image.

**FPGA → Computer**

1. FPGA receives the image.
2. UART controller starts TPU inference.
3. FPGA waits for inference to complete.
4. FPGA transmits one byte containing the predicted class index.

Current UART configuration:

- Baud rate: 921600
- Packet header: `0xAA`
- Image size: 784 bytes
- Return value: 1-byte class index
- Serial format: 8N1

## Software

The `model/` directory contains the model, Python application, and supporting scripts.

Important files:

- `quickdraw_canvas_predict.py`: Drawing interface and local/FPGA inference.
- `quickdraw_dense_32_int8.tflite`: Quantized TensorFlow Lite model.
- `quickdraw_dense_32.keras`: Original Keras model.
- `model.py`: Model training.
- `extract_int8_weights.py`: Extracts quantized model parameters.
- `make_mem_files.py`: Generates FPGA memory initialization files.
- `validate_fpga_inference.py`: Validates FPGA inference results.

The Python application supports comparing the local TensorFlow Lite prediction with the FPGA prediction.

It also measures CPU inference latency, FPGA inference latency, UART transmission time, and total FPGA transaction time separately.

## Performance Measurements

The following measurements were obtained from the current hardware implementation.

### Single-Inference Comparison

| Measurement | Result |
|---|---:|
| CPU inference (Intel Core Ultra 9) | 0.148 ms |
| FPGA inference (Artix-7) | 1.34877 ms |
| FPGA clock frequency | 100 MHz |
| FPGA inference cycles | 134,877 |
| CPU relative speed | Approximately 9.1× faster |

The CPU measurement represents one complete TensorFlow Lite prediction, including input tensor transfer, model invocation, output retrieval, and class selection.

The FPGA measurement represents the accelerator's execution interval, measured using the Integrated Logic Analyzer (ILA).

CPU model initialization and FPGA configuration are excluded from the per-inference comparison.

### UART and End-to-End Timing

A representative hardware benchmark produced:

| Measurement | Result |
|---|---:|
| UART write | 0.396 ms |
| UART flush | 0.042 ms |
| FPGA response wait | 10.462 ms |
| FPGA transaction | 10.900 ms |
| Theoretical UART transfer time | 8.529 ms |

The theoretical UART transfer time is based on transmitting the image packet and receiving the prediction using 8N1 serial communication at 921600 baud.

The measured transaction time also includes USB-UART and host communication overhead.

### Performance Interpretation

The FPGA currently achieves approximately 1.35 ms of neural network inference latency.

Although the Intel Core Ultra 9 completes this particular small neural network faster, the FPGA demonstrates a functioning, dedicated neural network accelerator implemented entirely in custom hardware.

The performance comparison is specific to this model, FPGA platform, and implementation. It is not a general comparison of CPU and FPGA capabilities.

## FPGA Configuration and Flash Programming

The accelerator was implemented and tested on the Digilent Basys 3.

The FPGA was programmed through Vivado Hardware Manager during development.

A configuration memory file (`tpu_config.mcs`) was also generated from the hardware bitstream and successfully programmed into the board's onboard SPI configuration flash.

Flash programming was verified successfully through Vivado:

- Configuration format: MCS
- Flash interface: SPIx4
- Erase operation: Successful
- Program operation: Successful
- Verify operation: Successful

The onboard flash allows the FPGA to automatically load the TPU configuration when the board is powered on, without requiring a new JTAG programming operation each time.

## FPGA Platform

- **Board:** Digilent Basys 3
- **FPGA:** AMD/Xilinx Artix-7 XC7A35T
- **Clock frequency:** 100 MHz
- **Development environment:** Vivado 2026.1
- **HDL:** SystemVerilog
- **Configuration:** Onboard SPI flash

## Repository Structure

```text
riscv-tpu-accelerator/
│
├── ILA/
│   ├── ila_tpu_debug.xci
│   └── uart_tpu_system.ltx
│
├── constraints/
│   └── basys3.xdc
│
├── images/
│
├── mem_files/
│   ├── dense_layer_test/
│   └── output_layer_test/
│
├── model/
│   ├── quickdraw_canvas_predict.py
│   ├── quickdraw_dense_32.keras
│   ├── quickdraw_dense_32_int8.tflite
│   └── ...
│
├── rtl/
│   ├── dense_layer_controller.sv
│   ├── dense_tile_controller.sv
│   ├── requantize_int8.sv
│   ├── systolic_array_8x8.sv
│   ├── systolic_pe.sv
│   ├── tpu_top.sv
│   ├── uart_rx.sv
│   ├── uart_tx.sv
│   ├── uart_tpu_controller.sv
│   └── uart_tpu_system.sv
│
├── sim/
│   └── ...
│
├── tcl scripts/
│   └── copy_mem_files.tcl
│
└── README.md
```

## Future Work

Potential directions for future development include:

- Improve timing closure and maximum operating frequency.
- Explore more efficient systolic array architectures.
- Investigate alternative memory architectures and data movement.
- Explore integration with the custom RISC-V processor.
- Experiment with larger neural networks and additional hardware acceleration workloads.

## Project Status

The current implementation successfully performs INT8 neural network inference on the Basys 3 FPGA using real image data transmitted over UART.

The hardware has been tested using the Python application, benchmarked against local TensorFlow Lite inference, and programmed into the board's onboard configuration flash.

The eight-bank weight memory architecture reduces tile-loading cycles by 4.5× compared with the original sequential loading approach.

This establishes a working baseline for future TPU architectural experiments and optimization.
