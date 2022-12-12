#include "pulp_nnx_util.h"
#include "pulp_nnx_hal.h"

void nnx_activate_gvsoc_logging(int log_level) {
  NEUREKA_WRITE_IO_REG(NEUREKA_REG_GVSOC_TRACE, log_level);
}

void nnx_deactivate_gvsoc_logging() {
  NEUREKA_WRITE_IO_REG(NEUREKA_REG_GVSOC_TRACE, 0);
}

