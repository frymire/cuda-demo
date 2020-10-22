/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.
 * Any use, reproduction, disclosure, or distribution of this software
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA)
 * associated with this source code for terms and conditions that govern
 * your use of this NVIDIA software.
 *
 */

#include "../common/book.h"
#include "../common/cpu_bitmap.h"

const int dim = 1000;

struct cuComplex {
  float r;
  float i;
  cuComplex(float a, float b): r(a), i(b) {}
  float magnitude2(void) { return r*r + i*i; }
  cuComplex operator*(const cuComplex& a) { return cuComplex(r*a.r - i * a.i, i*a.r + r * a.i); }
  cuComplex operator+(const cuComplex& a) { return cuComplex(r + a.r, i + a.i); }
};


int julia(int x, int y);
void fillBitmapWithJuliaValues(unsigned char *bitmapData);

int main(void) {
  CPUBitmap bitmap(dim, dim);
  fillBitmapWithJuliaValues(bitmap.get_ptr());
  bitmap.display_and_exit();
}


void fillBitmapWithJuliaValues(unsigned char *bitmapData) {

  for (int y = 0; y < dim; y++) {
    for (int x = 0; x < dim; x++) {
      int offset = x + y*dim;
      bitmapData[offset*4 + 0] = 255*julia(x, y);
      bitmapData[offset*4 + 1] = 0;
      bitmapData[offset*4 + 2] = 0;
      bitmapData[offset*4 + 3] = 255;
    }
  }

}


int julia(int x, int y) {

  const float scale = 1.5;
  float jx = scale * (float) (dim/2 - x)/(dim/2);
  float jy = scale * (float) (dim/2 - y)/(dim/2);
  cuComplex a(jx, jy);

  cuComplex c(-0.8, 0.156);

  int i = 0;
  for (i = 0; i < 200; i++) {
    a = a*a + c;
    if (a.magnitude2() > 1000) return 0;
  }

  return 1;
}
