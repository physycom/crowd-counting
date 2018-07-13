#include "MRF_core.h"
#include <jsoncons/json.hpp>

using namespace std;

extern float DISC_K;      // truncation of discontinuity cost
extern float DATA_K;      // truncation of data cost
extern float LAMBDA;      // weighting of data cost

#define SW_VER     123

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
  auto pred = jpred["patch_count"].as<std::vector<float>>();
  cout << "Width  : " << width << endl;
  cout << "Height : " << height << endl;
  cout << "Size   : " << pred.size() << endl;

	image<uchar> *img = new image<uchar>(width+2, height+2);

  // fill the bulk
  for (int i = 0; i < width; ++i)
    for(int j = 0; j < height; ++j)
//      imRef(img, i, j) = pred[(j-1) + height*(i-1)];
      imRef(img, i+1, j+1) = uchar(pred[i + width*j] + 0.5f);
      //imRef(img, i, j) = pred[i + width*j];

  // fill the boundary
  for(int j = 1; j < height+1; ++j)
  {
    imRef(img, 0, j)       = imRef(img, 1, j);
    imRef(img, width+1, j) = imRef(img, width, j);
  }
  for (int i = 0; i < width+2; ++i)
  {
    imRef(img, i, 0)        = imRef(img, i, 1);
    imRef(img, i, height+1) = imRef(img, i, height);
  }

  DISC_K = 3500.0f;
  DATA_K = 1000.0f;
  LAMBDA = 0.85f;
  image<uchar> *out = restore_ms(img);

  int finalcount = 0;
  for (int i = 1; i < width+1; i+=2)
    for(int j = 1; j < height+1; j+=2)
      finalcount += imRef(out, i, j);
  if (width%2==0)
    for(int j = 1; j < height+1; j+=2)
      finalcount += uchar((imRef(out, width, j)/2.f) + 0.5f);
  if (height%2==0)
    for(int i = 1; i < width+1; i+=2)
      finalcount += uchar((imRef(out, i, height)/2.f) + 0.5f);

  cout << "Count  : " << finalcount << endl;

  string info_name = pred_file.substr(0,pred_file.find_first_of(".")) + ".json_physycom";
  cout << "Output : " << info_name << endl;
  string timestamp = "12345";
  string loctag = "fake";
  ofstream info_json(info_name);
  info_json << "{\n";
  info_json << "\t\"frame_00000\" : {\n";
  info_json << "\t\t\"timestamp\" : " << timestamp << ",\n";
  info_json << "\t\t\"id_box\" : \"location\",\n";
  info_json << "\t\t\"detection\" : \"crowd3\",\n";
  info_json << "\t\t\"sw_ver\" : " << SW_VER << ",\n";
  info_json << "\t\t\"people_count\" : [{\"id\" : \"" << loctag << "\", \"count\" : " << finalcount << "}],\n";
  info_json << "\t\t\"diagnostics\" : [{\"id\" : \"coming\", \"value\" : \"soon\"}]\n";
  info_json << "\t}\n";
  info_json << "}";
  info_json.close();

	return 0;
}