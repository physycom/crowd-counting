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

#define ITER 5      // number of BP iterations at each scale
#define LEVELS 5     // number of scales
#define VALUES 400   // number of possible graylevel values

constexpr float INF = std::numeric_limits<float>::infinity(); 

void msg(const float &s1[VALUES], const float &s2[VALUES],
         const float &s3[VALUES], const float &s4[VALUES],
         const float &dst[VALUES])
{
    for(int value = 0; value < VALUES; ++value) dst[value] = s1[value] + s2[value] + s3[value] + s4[value];
    float minimum = *std::min_element(dst, dts + VALUES),
          s,
          *z = new float[VALUES+1];
    int k = 0,
        *v = new float[VALUES];
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
                    return d - val;
                   });
    return;
}

cv::Mat MRF(const cv::Mat &img, const float *options)
{
    int height = img.size().height,
        width  = img.size().width,
        best,
        new_height, new_width;
    float best_val, val,
          DISC_K = options[0];// 3000.0; // truncation of discontinuity cost
          DATA_K = options[1];// 1000.0; // truncation of data cost
          LAMBDA = options[2];// 1.0; // weighting of data cost

    cv::Mat out(width, height, CV_32FC1);
    
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
                imRef(data[0], x, y)[value] = LAMBDA * std::min( (img.at<float>(x, y) - value)*(img.at<float>(x, y) - value) , DATA_K); 
    // ======================================================

    // data pyramid
#pragma omp parallel for private(new_width, new_height)
    for(int i = 1; i < LEVELS; ++i)
    {
        new_width = (int)std::ceil(data[i-1]->width() / 2.f);
        new_height = (int)std::ceil(data[i-1]->height() / 2.f);

#ifdef DEBUG
        assert(new_width >= 1);
        assert(new_height >= 1);
#endif
        data[i] = new image<float[VALUES]>(new_width, new_height);
        for (int y = 0; y < data[i-1]->height(); ++y)
            for (int x = 0; x < data[i-1]->width(); ++x)
                for (int value = 0; value < VALUES; ++value)
                    imRef(data[i], x/2, y/2)[value] += imRef(data[i-1], x, y)[value];

    }

    // belief propagation using checkerboard update scheme

    // run bp from coarse to fine
    new_width = data[LEVELS - 1]->width();
    new_height = data[LEVELS - 1]->height();
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
                msg(imRef(u, x, y+1),imRef(d, x, y-1),imRef(r, x-1, y), imRef(data, x, y), imRef(r, x, y));
                msg(imRef(u, x, y+1),imRef(d, x, y-1),imRef(l, x+1, y), imRef(data, x, y), imRef(l, x, y));
                //msg(u.at<float>(x, y+1), d.at<float>(x, y-1), r.at<float>(x-1, y), data.at<float>(x, y), r.at<float>(x, y));
                //msg(u.at<float>(x, y+1), d.at<float>(x, y-1), l.at<float>(x+1, y), data.at<float>(x, y), l.at<float>(x, y));
            }

    for(int i = LEVELS-2; i >= 0; --i)
    {
        new_width = data[i]->width();
        new_height = data[i]->height();
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
                    msg(imRef(u, x, y+1),imRef(d, x, y-1),imRef(r, x-1, y), imRef(data, x, y), imRef(r, x, y));
                    msg(imRef(u, x, y+1),imRef(d, x, y-1),imRef(l, x+1, y), imRef(data, x, y), imRef(l, x, y));
                    //msg(u.at<float>(x, y+1), d.at<float>(x, y-1), r.at<float>(x-1, y), data.at<float>(x, y), r.at<float>(x, y));
                    //msg(u.at<float>(x, y+1), d.at<float>(x, y-1), l.at<float>(x+1, y), data.at<float>(x, y), l.at<float>(x, y));
                }
    }

    // output function
#pragma omp parallel for private(best, best_val, val)
    for(int y = 1; y < height - 1; ++y)
        for(int x = 1; x < width - 1; ++x)
        {
            best = 0;
            best_val = INF;
            for(int value = 0; value < VALUES; ++values)
            {
                val = imRef(l, x+1, y)[value] + //l[0]->at<cv::Vec3b>(x+1, y)[value] +
                      imRef(r, x-1, y)[value] + //r[0]->at<cv::Vec3b>(x-1, y)[value] +
                      imRef(data, x,y)[value];  //data[0]->at<cv::Vec3b>(x,y)[value];
                if(val < best_val)
                {
                    best_val = val;
                    best = value;
                }
            }
            out.at<float>(x, y) = best;
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

float* Evaluate(float *MRFParams, Feature* features, const int &n)
{   
    float *finalcount = new float[n];
    cv::Mat p;
#pragma omp parallel for
    for(int i = 0; i < n; ++i)
    {


        // The marginal data of the predicted count matrix is 0 after apply MRF, 
        // so first extending the predicted count matrix by copy marginal data.


        // apply MRF
        p = MRF(MRFParams, p);
        // FinalCount function
        finalcount[i] = 0.f;
        for(int i = 0; i < Nrow; ++i)
            for(int j = 0; j < Ncol; ++j)
                finalcount[i] += (j == Ncol - 1 && !(Nrow % 2)) ? p[i][j] * .5f : 
                                 (i == Nrow - 1 && !(Ncol % 2)) ? p[i][j] * .5f : /*WRONG???*/
                                 (i % 2) ? 0.f :
                                 (j % 2) ? 0.f :
                                 p[i][j];
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
        if(im_r.channels() == 1)
        {
            std::vector<cv::Mat> channels(3);
            channels[0] = im_r;
            channels[1] = im_r;
            channels[2] = im_r;
            cv::merge(channels, im_r);
        }

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

