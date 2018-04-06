#if defined (_WIN32)
const char sep = '\\';
#define GetCurrentDir _getcwd
#define Popen _popen
#include <Windows.h>
#include <direct.h>
#else
const char sep = '/';
#define GetCurrentDir getcwd
#define Popen popen
#include <dirent.h>
#include <sys/types.h>
#include <errno.h>
#include <unistd.h>
#include <stdio.h>
#endif

#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <climits>

#include "my_image.hpp" // (TODO) remove and use opencv
#include <opencv/opencv.hpp>

#ifdef DEBUG
#include <assert.h>
#endif

#define MULTI_CHANNELS
#define ITER 5      // number of BP iterations at each scale
#define LEVELS 5     // number of scales
#define VALUES 400   // number of possible graylevel values

constexpr float INF = std::numeric_limits<float>::infinity(); 

void msg(const float s1[VALUES], const float s2[VALUES],
         const float s3[VALUES], const float s4[VALUES],
         float dst[VALUES], const float DISC_K)
{
    for(int value = 0; value < VALUES; ++value) dst[value] = s1[value] + s2[value] + s3[value] + s4[value];
    float minimum = *std::min_element(dst, dst + VALUES),
          s,
          *z = new float[VALUES+1];
    int k = 0,
        *v = new int[VALUES];
    // dt function
    for(int q = 1; q <= VALUES-1; ++q)
    {
        s = ((dst[q] + q*q) - (dst[v[k]] + v[k]*v[k])) / (2*(q - v[k]));
        while (s <= z[k--])
            s = ((dst[q] + q*q) - (dst[v[k]] + v[k]*v[k])) / (2*(q - v[k]));
        ++k;
        v[k] = q;
        z[k] = s;
        z[k+1] = INF;
    }
    k = 0;
    minimum += DISC_K;
    for(int q = 0; q <= VALUES-1; ++q)
    {
        while (z[k+1] < q) ++k;
        // truncate and store in destination vector
        dst[q] = std::min((q - v[k])*(q - v[k]) + dst[v[k]], minimum);
    }
    delete[] z;
    delete[] v;
    // normalize
    s = std::accumulate(dst, dst + VALUES, 0.f) / VALUES;
    std::transform(dst, dst + VALUES, dst,
                   [&s](const float &d)
                   {
                    return d - s;
                   });
    return;
}

int* MRF(const int *img, const int &height, const int &width, const float *options)
{
    int best,
        new_height, new_width,
        *out = new int[width*height];
    float best_val, val,
          DISC_K = options[0],// 3000.0; // truncation of discontinuity cost
          DATA_K = options[1],// 1000.0; // truncation of data cost
          LAMBDA = options[2];// 1.0; // weighting of data cost
    
    image<float[VALUES]> *u[LEVELS]; // up
    image<float[VALUES]> *d[LEVELS]; // down
    image<float[VALUES]> *l[LEVELS]; // left
    image<float[VALUES]> *r[LEVELS]; // right
    image<float[VALUES]> *data[LEVELS]; // data
    // COMP_DATA
    data[0] = new image<float[VALUES]>(width, height);
    // ======================================================
    for(int y = 0; y < height; ++y)
        for(int x = 0; x < width; ++x)
            for(int value = 0; value < VALUES; ++value)
                imRef(data[0], x, y)[value] = LAMBDA * std::min( float(img[x*width + y] - value)*(img[x*width + y] - value), DATA_K); 
    // ======================================================

    // data pyramid
#pragma omp parallel for private(new_width, new_height)
    for(int i = 1; i < LEVELS; ++i)
    {
        new_width = (int)std::ceil(data[i-1]->width / 2.f);
        new_height = (int)std::ceil(data[i-1]->height / 2.f);

#ifdef DEBUG
        assert(new_width >= 1);
        assert(new_height >= 1);
#endif
        data[i] = new image<float[VALUES]>(new_width, new_height);
        for (int y = 0; y < data[i-1]->height; ++y)
            for (int x = 0; x < data[i-1]->width; ++x)
                for (int value = 0; value < VALUES; ++value)
                    imRef(data[i], x/2, y/2)[value] += imRef(data[i-1], x, y)[value];

    }

    // belief propagation using checkerboard update scheme

    // run bp from coarse to fine
    new_width = data[LEVELS - 1]->width;
    new_height = data[LEVELS - 1]->height;
    // in the coarsest level messages are initialized to zero
    u[LEVELS - 1] = new image<float[VALUES]>(new_width, new_height);
    d[LEVELS - 1] = new image<float[VALUES]>(new_width, new_height);
    l[LEVELS - 1] = new image<float[VALUES]>(new_width, new_height);
    r[LEVELS - 1] = new image<float[VALUES]>(new_width, new_height);
    // FIRST STEP of BP
    for(int t = 0; t < ITER; ++t)
        for(int y = 1; y < new_height - 1; ++y)
            for(int x = ((y + t) % 2) + 1; x < new_width - 1; x += 2)
            {
                msg(imRef(u[LEVELS - 1], x, y+1), imRef(d[LEVELS - 1], x, y-1), imRef(r[LEVELS - 1], x-1, y), imRef(data[LEVELS - 1], x, y), imRef(r[LEVELS - 1], x, y), DISC_K);
                msg(imRef(u[LEVELS - 1], x, y+1), imRef(d[LEVELS - 1], x, y-1), imRef(l[LEVELS - 1], x+1, y), imRef(data[LEVELS - 1], x, y), imRef(l[LEVELS - 1], x, y), DISC_K);
                //msg(u.at<float>(x, y+1), d.at<float>(x, y-1), r.at<float>(x-1, y), data.at<float>(x, y), r.at<float>(x, y), DISC_K);
                //msg(u.at<float>(x, y+1), d.at<float>(x, y-1), l.at<float>(x+1, y), data.at<float>(x, y), l.at<float>(x, y), DISC_K);
            }

    for(int i = LEVELS-2; i >= 0; --i)
    {
        new_width = data[i]->width;
        new_height = data[i]->height;
        // initialize messages from values of previous level
        u[i] = new image<float[VALUES]>(width, height, false);
        d[i] = new image<float[VALUES]>(width, height, false);
        l[i] = new image<float[VALUES]>(width, height, false);
        r[i] = new image<float[VALUES]>(width, height, false);

        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
                for (int value = 0; value < VALUES; value++)
                {
                    imRef(u[i], x, y)[value] = imRef(u[i+1], x/2, y/2)[value];
                    imRef(d[i], x, y)[value] = imRef(d[i+1], x/2, y/2)[value];
                    imRef(l[i], x, y)[value] = imRef(l[i+1], x/2, y/2)[value];
                    imRef(r[i], x, y)[value] = imRef(r[i+1], x/2, y/2)[value];
                }
        // delete old messages and data
        delete u[i+1];
        delete d[i+1];
        delete l[i+1];
        delete r[i+1];
        delete data[i+1];

        // BP
        for(int t = 0; t < ITER; ++t)
            for(int y = 1; y < new_height - 1; ++y)
                for(int x = ((y + t) % 2) + 1; x < new_width - 1; x += 2)
                {
                    msg(imRef(u[i], x, y + 1), imRef(d[i], x, y - 1), imRef(r[i], x - 1, y), imRef(data[i], x, y), imRef(r[i], x, y), DISC_K);
                    msg(imRef(u[i], x, y + 1), imRef(d[i], x, y - 1), imRef(l[i], x + 1, y), imRef(data[i], x, y), imRef(l[i], x, y), DISC_K);
                    //msg(u.at<float>(x, y+1), d.at<float>(x, y-1), r.at<float>(x-1, y), data.at<float>(x, y), r.at<float>(x, y), DISC_K);
                    //msg(u.at<float>(x, y+1), d.at<float>(x, y-1), l.at<float>(x+1, y), data.at<float>(x, y), l.at<float>(x, y), DISC_K);
                }
    }

    // output function
#pragma omp parallel for private(best, best_val, val)
    for(int y = 1; y < height - 1; ++y)
        for(int x = 1; x < width - 1; ++x)
        {
            best = 0;
            best_val = INF;
            for(int value = 0; value < VALUES; ++value)
            {
                val = imRef(l[0], x+1, y)[value] + //l[0]->at<cv::Vec3b>(x+1, y)[value] +
                      imRef(r[0], x-1, y)[value] + //r[0]->at<cv::Vec3b>(x-1, y)[value] +
                      imRef(data[0], x,y)[value];  //data[0]->at<cv::Vec3b>(x,y)[value];
                if(val < best_val)
                {
                    best_val = val;
                    best = value;
                }
            }
            out[x*width + y] = best;
        }

    delete u[0];
    delete d[0];
    delete l[0];
    delete r[0];
    delete data[0];

    return out;
}


class Feature
{
    int size;
    float **feature;
public:
    int n;
    Feature(const int &h, const int &w, const int &n)
    {
        // every row is a patch
        this->feature = new float*[h*w];
        std::generate(this->feature, this->feature + h*w, [&n](){return new float[n];});
        this->h = h;
        this->w = w;
        this->n = n;
    }

    ~Feature()
    {
        for(int i = 0; i < this->h*this->w; ++i) delete[] this->feature[i];
        delete[] this->feature;
    }

    inline float* operator[](const int &i, const int &j)
    {
        return this->feature[i*this->w + j];
    }

    inline int size()
    {
        return this->h * this->w;
    }
};

void dumpFeature(const std::string &filename, Feature *feature, const int &n)
{
    std::ofstream os(filename + ".info");
    os  << filename << " INFO (number of images, features size)" << std::endl
        << n << std::endl;
    for(int i = 0; i < n; ++i) os << feature[i].size() << std::endl;
    os.close();

#ifdef BINOUT
    os.open(filename, std::ios::out | std::ios::binary);
    for(int i = 0; i < n; ++i)
        for(int j = 0; j < feature[i].size(); ++j)
            for(int k = 0; k < feature[i].n; ++k)
                os.write( (const char *) &feature[i][j][k], sizeof(float));
    os.close();
#else
    os.open(filename);
    for(int i = 0; i < n; ++i)
        for(int j = 0; j < feature[i].size(); ++j)
        {
            for(int k = 0; k < feature[i].n; ++k)
                os << feature[i][j][k] << ",";
            os << std::endl;
        }
#endif
    return;
}

float* Evaluate(const float* predictions, 
                const int *width, 
                const int *height, 
                const int &n, 
                const float *MRFParams)//, const float *ground)
{   
    float *finalcount = new float[n],
          *p = nullptr,
          *tmp = nullptr;
    int k = 0;

#pragma omp parallel for reduction(+:k)
    for(int i = 0; i < n; ++i) // loop over patches
    {
        // The marginal data of the predicted count matrix is 0 after apply MRF, 
        // so first extending the predicted count matrix by copy marginal data.
        p = new float[height[i] * width[i]];
        // compute the transpose
        for(int ii = 0; ii < height[i]; ++ii)
            for(int jj = 0; jj < width[i]; ++jj)
                p[jj*height[i] + ii] = predictions[k + ii*width[i] + jj];
        std::memcpy(p, predictions + k, sizeof(float)*height[i]*width[i]);
        // miss add border

        k += height[i] * width[i];
        // apply MRF
        //p = MRF(p, height[i], width[i], MRFParams);
        // FinalCount function
        finalcount[i] = 0.f;
        for(int j = 0; j < height[i]; ++j)
            for(int k = 0; k < width[i]; ++k)
                finalcount[i] += (k == width[i]  - 1 && !(height[i] % 2)) ? p[j*width[i] + k] * .5f : 
                                 (j == height[i] - 1 && !(width[i] % 2))  ? p[j*width[i] + k] * .5f :
                                 (j % 2) ? 0.f :
                                 (k % 2) ? 0.f :
                                 p[j*width[i] + k];
        delete[] p;
    }
    //float MAE = std::inner_product(finalcount, finalcount + n, ground,
    //                               0.f, std::plus<float>(),
    //                              [](const float &fc, const float &gt)
    //                              {
    //                                  return std::abs(fc - gt);
    //                              }) / n,
    //    MSE = std::inner_product(finalcount, finalcount + n, ground,
    //                              0.f, std::plus<float>(), 
    //                              [](const float &fc, const float &gt)
    //                              {
    //                                  return std::sqrt((fc - gt) * (fc - gt));
    //                              }) / n;
    //std::cout << "MAE: " << MAE << std::endl
    //          << "MSE: " << MSE << std::endl;
    return finalcount;
}

inline std::vector<std::string> files_in_directory(const std::string &directory)
    {
        std::vector<std::string> files;
#if defined (_WIN32)
        WIN32_FIND_DATA fileData;
        HANDLE hFind;
        if (!((hFind = FindFirstFile((directory + "*").c_str(), &fileData)) == INVALID_HANDLE_VALUE))
            while (FindNextFile(hFind, &fileData))
                if (file_exists(directory + fileData.cFileName))
                    files.push_back(fileData.cFileName);

        FindClose(hFind);
#else
        DIR *dp;
        struct dirent *dirp;
        if((dp  = opendir(directory.c_str())) == nullptr)
        {
            std::cerr << "Error(" << errno << ") opening " << directory << std::endl;
            exit(1);
        }
        while ((dirp = readdir(dp)) != nullptr)
            if(directory + std::string(dirp->d_name) != "." && directory + std::string(dirp->d_name) != ".." && !dir_exists(dirp->d_name))
                files.push_back(std::string(dirp->d_name));
        closedir(dp);
#endif
        return files;
    }


void ExtractFeatures(const std::string &imagesPath, 
                     std::string wNN = "imagenet-resnet-152-dag.mat", 
                     std::string mode = "test", 
                     bool memory = false
                     )
{
    std::vector<std::string> img_names = files_in_directory(imagesPath);
    int n = (int)img_names.size(), // images number
        winSize = 100, winStep = winSize - 1,
        heightStep = 50, widthStep = 50;
    Feature *features = new Feature[n];

    // miss import weights NN

    // init pre-trained resnet-152 model
#ifdef MULTI_IMAGE
#pragma omp parallel for
#endif
    for(int i = 0; i < n; ++i)
    {
#ifdef DEBUG
        std::cout << "Processing images #" << i << " ('" << img_names[i] << "')" << std::endl;
#endif
        cv::Mat im = cv::imread(img_names[i]);
        cv::Size newSize((im.size().height / heightStep) * heightStep, 
                         (im.size().width / widthStep) * widthStep
                         );

        cv::Mat im_r; // dst image
        cv::resize(im, im_r, newSize); //resize image
        im.release();
#ifndef MULTI_CHANNELS
        if(im_r.channels() == 1)
        {
            std::vector<cv::Mat> channels(3);
            channels[0] = im_r;
            channels[1] = im_r;
            channels[2] = im_r;
            cv::merge(channels, im_r);
        }
#endif
        features[i] = Feature(newHeight / 50 - 1, newWindth / 50 - 1, 1000);
#ifndef MULTI_IMAGE
#pragma omp parallel for
#endif
        for(int y = 0, row = 0; y <= newWindth; y += winStep, ++row)
            for(int x = 0, column = 0; x <= newWindth; x += winStep, ++column)
            {

                features[i][row, column] = ;
            }
    }

    dumpFeature("phase_0.csv", features, n);
    delete[] features;

    return;
}

