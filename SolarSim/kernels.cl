#define SIZE 100000
#define HALF 5000
#define MASS 0
#define X 1
#define Y 2
#define DX 3
#define DY 4
#define FX 5
#define FY 6
#define R 7
#define PLANET_SIZE 8
#define VEC_WIDTH 8
#define SIMD_IDX(a,b) ((a / VEC_WIDTH)*(VEC_WIDTH*PLANET_SIZE)+(a % VEC_WIDTH)+VEC_WIDTH*b)
#define G 4.0

__kernel void planetForce(__global float planet[SIZE*PLANET_SIZE])
{
	int i = get_global_id(0);
	if(i<HALF)return;
	for(int j=0;j<SIZE;j++){
		float F,rx,ry;
		rx = planet[SIMD_IDX(i,X)]-planet[SIMD_IDX(j,X)];
		ry = planet[SIMD_IDX(i,Y)]-planet[SIMD_IDX(j,Y)];
		F = (planet[SIMD_IDX(i,MASS)]*planet[SIMD_IDX(j,MASS)]*G)/((rx*rx)+(ry*ry)+0.00000000000001f);
		planet[SIMD_IDX(i,FX)] += (-F*rx);
		planet[SIMD_IDX(i,FY)] += (-F*ry);
	}
}