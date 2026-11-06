# RISC-V TPU Accelerator

![Neural Network Accelerator Architecture](images/Neural_Network_Accelerator_Architecture_Infographic.png)

An INT8 neural network accelerator implemented in SystemVerilog on a **Digilent Basys 3 FPGA**. The project combines a custom 8×8 systolic array, dense-layer controllers, on-chip memory, and UART communication to perform neural network inference on hardware.

The goal is to explore how specialized hardware accelerates neural network workloads compared with running the same model in software.

## Project Overview

The accelerator runs a quantized neural network using an 8-bit integer datapath.

The current network architecture consists of:

- **Input:** 784 values (28×28 image)
- **Hidden layer:** 128 neurons
- **Output layer:** 32 classes
- **Arithmetic:** INT8 weights and activations with wider accumulators
- **Compute engine:** Shared 8×8 systolic array

A Python application allows a user to draw an image, run local inference using TensorFlow Lite, and send the same image to the FPGA for hardware inference.

The FPGA returns its predicted class over UART.

## Hardware Architecture

The main hardware components are:

| Component | Purpose |
|---|---|
| `tpu_top.sv` | Top-level neural network accelerator |
| `systolic_array_8x8.sv` | 8×8 parallel multiply-accumulate array |
| `systolic_pe.sv` | Individual systolic processing element |
| `dense_layer_controller.sv` | Controls dense-layer computation |
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

### Numeric Representation

The network uses INT8 quantized weights and activations. Accumulation uses wider arithmetic to preserve intermediate results before requantization.

Weights, biases, multipliers, and shift values are initialized using `.mem` files.

## Computer-to-FPGA Communication

The computer communicates with the Basys 3 through USB-UART.

The current communication protocol is:

**Computer → FPGA**
1. Send packet header: `0xAA`
2. Send 784 image bytes, representing a 28×28 image.

**FPGA → Computer**
1. FPGA receives the image.
2. UART controller starts TPU inference.
3. FPGA waits for inference to complete.
4. FPGA transmits one byte containing the predicted class index.

Current UART configuration:
- Baud rate: 115200
- Packet header: `0xAA`
- Image size: 784 bytes
- Return value: 1-byte class index

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

## Performance Measurements

The following measurements were obtained during development:

| Measurement | Result |
|---|---:|
| Local TensorFlow Lite inference | 0.125 ms |
| FPGA end-to-end response | 74.171 ms |
| Estimated UART transfer time | 68.229 ms |
| Estimated FPGA compute time from end-to-end timing | 5.942 ms |
| RTL cycle-count inference time | 4.08806 ms |

The RTL cycle-count measurement used:

- Clock frequency: 100 MHz
- Clock cycles: 408,806
- Calculated inference time: 4.08806 ms

The UART estimate is based on transmitting the image packet at 115200 baud. The end-to-end measurement includes communication overhead, while the RTL cycle count measures the accelerator's simulated execution interval.

These measurements are from the current implementation and should not be interpreted as a direct comparison of optimized CPU and FPGA hardware.

## FPGA Platform

- Board: Digilent Basys 3
- FPGA: AMD/Xilinx Artix-7 XC7A35T
- Clock frequency: 100 MHz
- Development environment: Vivado 2026.1
- HDL: SystemVerilog

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

- Optimize systolic array utilization and inference latency.
- Reduce UART communication overhead.
- Explore integration with the custom RISC-V processor.
- Investigate more efficient memory architectures.
- Expand hardware acceleration capabilities.
