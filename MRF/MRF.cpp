/******************************************************************************
An MRF implementation for the paper
@article{han2017image,
  title={Image Crowd Counting Using Convolutional Neural Network and Markov Random Field},
  author={Han, Kang and Wan, Wanggen and Yao, Haiyan and Hou, Li},
  journal={arXiv preprint arXiv:1706.03686},
  year={2017}
}

This code is modified from the following paper
@inproceedings{Felzenszwalb2004Efficient,
  title={Efficient belief propagation for early vision},
  author={Felzenszwalb, P. F. and Huttenlocher, D. R.},
  booktitle={Computer Vision and Pattern Recognition, 2004. CVPR 2004. Proceedings of the 2004 IEEE Computer Society Conference on},
  pages={I-261-I-268 Vol.1},
  year={2004},
}

******************************************************************************/
#include <iostream>
#include <math.h>
#include <assert.h>
#include <cstring>
#include <algorithm>
#include "mex.h"
#include "image.h"
#include "misc.h"
#include "MRF_core.h"

using namespace std;

extern float DISC_K;      // truncation of discontinuity cost
extern float DATA_K;      // truncation of data cost
extern float LAMBDA;      // weighting of data cost

void mexFunction(
        int          nlhs,
        mxArray      *plhs[],
        int          nrhs,
        const mxArray *prhs[]
        )
{
    if (nrhs != 2) {
        mexErrMsgIdAndTxt("MATLAB:mexcpp:nargin",
                "MEXCPP requires two input arguments.");
    } else if (nlhs > 1) {
        mexErrMsgIdAndTxt("MATLAB:mexcpp:nargout",
                "MEXCPP requires one output argument.");
    }
    int height = mxGetM(prhs[0]);
    int width = mxGetN(prhs[0]);
    uchar *p;
    p = (uchar*) mxGetData(prhs[0]);
    image<uchar> *img = new image<uchar>(width, height);
    //copy data to img
    for (int i = 0; i < width; i++){
        for(int j = 0; j < height; j++){
            imRef(img, i, j) = *p;
            p++;
        }
    }
    float *options = (float*) mxGetData(prhs[1]);
    DISC_K = *options++;
    DATA_K = *options++;
    LAMBDA = *options;

    image<uchar> *out = restore_ms(img);

    //copy data to result
    uchar *result;
    plhs[0] = mxCreateNumericMatrix(height, width, mxUINT8_CLASS, mxREAL);
    result = (uchar*) mxGetPr(plhs[0]);
    for (int i = 0; i < width; i++){
        for(int j = 0; j < height; j++){
            *result =  imRef(out, i, j);
            result++;
        }
    }
}
