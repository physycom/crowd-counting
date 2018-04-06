/*
Copyright (C) 2006 Pedro Felzenszwalb

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
*/

/* a simple image class */

#ifndef IMAGE_H
#define IMAGE_H
#include <cstring>

template <class T>
class image {
  /* image data. */
  T *data;
  
  /* init an image */
  
  void init(const T &val);

  /* copy an image */
  image<T> *copy() const;
public:
  /* raw pointers. */
  T **access;

  /* create an image */
  image(const int &width, const int &height, const bool init = true);

  /* delete an image */
  ~image();

  int width, height;
};

/* use imRef to access image data. */
#define imRef(im, x, y) (im->access[y][x])
  
/* use imPtr to get pointer to image data. */
#define imPtr(im, x, y) &(im->access[y][x])

template <class T>
image<T>::image(const int &width, const int &height, const bool init) {
  this->width = width;
  this->height = height;
  this->data = new T[this->width * this->height];  // allocate space for image data
  this->access = new T*[this->height];   // allocate space for row pointers
  
  // initialize row pointers
  for (int i = 0; i < this->height; i++) this->access[i] = this->data + (i * this->width);  
  if(init) std::memset(this->data, 0, this->width*this->height*sizeof(T));
}

template <class T>
image<T>::~image() {
  delete [] this->data; 
  delete [] this->access;
}

template <class T>
void image<T>::init(const T &val) {
  T *ptr = imPtr(this, 0, 0);
  T *end = imPtr(this, this->width - 1, this->height - 1);
  while (ptr <= end) *ptr++ = val;
}


template <class T>
image<T> *image<T>::copy() const {
  image<T> *im = new image<T>(this->width, this->height, false);
  std::memcpy(im->data, this->data, this->width * this->height * sizeof(T));
  return im;
}

#endif
  
