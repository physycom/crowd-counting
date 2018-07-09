#include "MRF_core.h"

int main()
{
	image<uchar> *img = new image<uchar>(100, 100);
	image<uchar> *out = restore_ms(img);

	return 0;
}