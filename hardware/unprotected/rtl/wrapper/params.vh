`ifndef __frodo_kem_vh__
`define __frodo_kem_vh__

// TMVP Parameters
`define TMVP_N  512
`define TMVP_TILE_SIZE 16
`define TMVP_DATA_WIDTH 8
// CAUTION! In usb_reg.v, cdc_fg_fifo gets data packets from CW as {8-bit data, bytecnt} and passes it
// in raw to RAMs during USB write process. If TMVP_DATA_WIDTH is changed, additional
// translation is needed between cdc_fg_fifo and RAM address resolution, as well as in dut_output_fifo.
`define TMVP_REAL_N 509

// Chipwhisperer Parameters
`define pDONE_EDGE_SENSITIVE  1
`define pADDR_WIDTH           21
`define pREG_RDDLY_LEN        3
`define pSYNC_STAGES          2

// Chipwhisperer Registers
`define pBYTECNT_SIZE       9 // maximum $clog2(N), depending on addr input for f_RAM in usb_reg
`define REG_DUT_GO          'h05
`define REG_DUT_FLUSHFIFO   'h06
`define REG_DUT_RESET       'h07
`define REG_DUT_RAM_F       'h08
`define REG_DUT_RAM_G       'h09
`define REG_DUT_DATA_OUT    'h0a


`endif