#ifndef CUSP_H
#define CUSP_H 
#include <glog/logging.h>
#include "common.h"
#include "DataLoader.cuh"
#define T float
//namespace cuSPARSE{

int run1(DataLoader& input, Metrics& metric);

int run2(DataLoader& input, Metrics& metric);

//}


#endif /* CUSP_H */
