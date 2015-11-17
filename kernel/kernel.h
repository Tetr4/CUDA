#ifndef KERNEL_H
#define KERNEL_H

void doCUDACalc(signed char *lb, signed char *rb, int zeilen, int spalten,int tauMax, int *profile,
 unsigned char *valid, int b, int h, bool useS, bool useF, int s);

#endif
