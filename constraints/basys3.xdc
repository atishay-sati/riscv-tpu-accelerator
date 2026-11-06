
## ============================================================
## BASYS 3 CONSTRAINT FILE
## ============================================================

## ============================================================
## 100 MHz ONBOARD CLOCK
## ============================================================

set_property PACKAGE_PIN W5 [get_ports clk]
set_property IOSTANDARD LVCMOS33 [get_ports clk]
create_clock -period 10.000 -name sys_clk [get_ports clk]


## ============================================================
## RESET BUTTON
## ============================================================

## Center button (BTNC), active high.
set_property PACKAGE_PIN U18 [get_ports reset]
set_property IOSTANDARD LVCMOS33 [get_ports reset]


## ============================================================
## USB UART COMMUNICATION
## ============================================================

## USB-UART receive: computer sends data into FPGA.
set_property PACKAGE_PIN B18 [get_ports uart_rx_pin]
set_property IOSTANDARD LVCMOS33 [get_ports uart_rx_pin]

## USB-UART transmit: FPGA sends predictions to computer.
set_property PACKAGE_PIN A18 [get_ports uart_tx_pin]
set_property IOSTANDARD LVCMOS33 [get_ports uart_tx_pin]


## ============================================================
## PREDICTED CLASS: LEDS 0-4
## ============================================================

set_property PACKAGE_PIN U16 [get_ports {predicted_class[0]}]
set_property PACKAGE_PIN E19 [get_ports {predicted_class[1]}]
set_property PACKAGE_PIN U19 [get_ports {predicted_class[2]}]
set_property PACKAGE_PIN V19 [get_ports {predicted_class[3]}]
set_property PACKAGE_PIN W18 [get_ports {predicted_class[4]}]

set_property IOSTANDARD LVCMOS33 [get_ports {predicted_class[0]}]
set_property IOSTANDARD LVCMOS33 [get_ports {predicted_class[1]}]
set_property IOSTANDARD LVCMOS33 [get_ports {predicted_class[2]}]
set_property IOSTANDARD LVCMOS33 [get_ports {predicted_class[3]}]
set_property IOSTANDARD LVCMOS33 [get_ports {predicted_class[4]}]


## ============================================================
## SYSTEM STATUS LEDS
## ============================================================

## LED 5: UART controller busy
set_property PACKAGE_PIN U15 [get_ports controller_busy]

## LED 6: TPU busy
set_property PACKAGE_PIN U14 [get_ports tpu_busy]

## LED 7: Packet completed
set_property PACKAGE_PIN V14 [get_ports packet_done]

## LED 8: Packet error
set_property PACKAGE_PIN V13 [get_ports packet_error]

set_property IOSTANDARD LVCMOS33 [get_ports controller_busy]
set_property IOSTANDARD LVCMOS33 [get_ports tpu_busy]
set_property IOSTANDARD LVCMOS33 [get_ports packet_done]
set_property IOSTANDARD LVCMOS33 [get_ports packet_error]

## ============================================================
## FPGA CONFIGURATION VOLTAGE
## ============================================================

set_property CFGBVS VCCO [current_design]
set_property CONFIG_VOLTAGE 3.3 [current_design]
