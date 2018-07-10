#include "MRF_core.h"
#include <jsoncons/json.hpp>

using namespace std;

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

	image<uchar> *img = new image<uchar>(width, height);

  for (int i = 0; i < width; ++i)
    for(int j = 0; j < height; ++j)
      imRef(img, i, j) = pred[i + width*j];

  cout << "Predictions" << endl;
  for(int j = 0; j < height; ++j)
  {
    for (int i = 0; i < width; ++i)
      //cout << pred[i + height*j] << " ";
      cout << int(imRef(img, i, j)) << " ";
    cout << endl;
  }

  image<uchar> *out = restore_ms(img);

  cout << "MRF" << endl;
  for(int j = 0; j < height; ++j)
  {
    for (int i = 0; i < width; ++i)
      cout << int(imRef(out, i, j)) << " ";
    cout << endl;
  }

	return 0;
}