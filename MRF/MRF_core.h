#include <iostream>
#include <algorithm>
#include "image.h"
#include "misc.h"

#define ITER        5       // number of BP iterations at each scale
#define LEVELS      5       // number of scales

#define INF         1E10    // large cost
#define VALUES      400     // number of possible graylevel values

static float *dt(float *f, int n);
void msg(float s1[VALUES], float s2[VALUES], float s3[VALUES], float s4[VALUES], float dst[VALUES]);
image<float[VALUES]> *comp_data(image<uchar> *img);
image<uchar> *output(image<float[VALUES]> *u, image<float[VALUES]> *d, image<float[VALUES]> *l, image<float[VALUES]> *r, image<float[VALUES]> *data);
void bp_cb(image<float[VALUES]> *u, image<float[VALUES]> *d, image<float[VALUES]> *l, image<float[VALUES]> *r, image<float[VALUES]> *data, int iter);
image<uchar> *restore_ms(image<uchar> *img);
