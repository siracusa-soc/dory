#include "pulp_nnx_util.h"
#include "pulp_nnx_hal.h"

void nnx_activate_gvsoc_logging(int use_dec) {
  NE16_WRITE_IO_REG(sizeof(nnx_task_t), 3);
  if(!use_dec) {
    NE16_WRITE_IO_REG(sizeof(nnx_task_t)+4, 3);
  }
  else{
    NE16_WRITE_IO_REG(sizeof(nnx_task_t)+4, 0);
  }
}

void ne16_deactivate_gvsoc_logging() {
  NE16_WRITE_IO_REG(sizeof(nnx_task_t), 0);
}

