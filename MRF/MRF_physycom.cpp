#include "MRF_core.h"
#include <jsoncons/json.hpp>

using namespace std;

extern float DISC_K;      // truncation of discontinuity cost
extern float DATA_K;      // truncation of data cost
extern float LAMBDA;      // weighting of data cost

int main(int argc, char **argv)
{
  std::string pred_file;
  if(argc != 2)
  {
    cerr << "Usage : " << argv[0] << " path/to/phase1/json" << endl;
    exit(1);
  }
  else
  {
    pred_file = argv[1];
  }

  jsoncons::json jpred = jsoncons::json::parse_file(pred_file);

  int width = jpred["width"].as<int>();
  int height = jpred["height"].as<int>();
  auto pred = jpred["patch_count"].as<std::vector<int>>();
  cout << "Width  : " << width << endl;
  cout << "Height : " << height << endl;
  cout << "Size   : " << pred.size() << endl;

	image<uchar> *img = new image<uchar>(width+2, height+2);

  // fill the bulk
  for (int i = 1; i < width-1; ++i)
    for(int j = 1; j < height-1; ++j)
      imRef(img, i, j) = pred[i + width*j];

  // fill the boundary
  for (int i = 1; i < width-1; ++i)
  {
    imRef(img, i, 0)        = imRef(img, i, 1);
    imRef(img, i, height-1) = imRef(img, i, height-2);
  }
  for(int j = 0; j < height; ++j)
  {
    imRef(img, 0, j)       = imRef(img, 1, j);
    imRef(img, width-1, j) = imRef(img, width-2, j);
  }

//  cout << "Predictions" << endl;
//  for(int j = 0; j < height; ++j)
//  {
//    for (int i = 0; i < width; ++i)
//      cout << int(imRef(img, i, j)) << " ";
//    cout << endl;
//  }

  DISC_K = 200.0;
  DATA_K = 200.0;
  LAMBDA = 1.0;
  image<uchar> *out = restore_ms(img);

//  cout << "MRF" << endl;
//  for(int j = 0; j < height; ++j)
//  {
//    for (int i = 0; i < width; ++i)
//      cout << int(imRef(out, i, j)) << " ";
//    cout << endl;
//  }


  int finalcount = 0;
  for (int i = 1; i < width-2; i+=2)
    for(int j = 1; j < height-2; j+=2)
      finalcount += imRef(out, i, j);
  cout << "Final count : " << finalcount << endl;



	return 0;
}