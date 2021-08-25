#define SIZE 10000
#define MASS 0
#define X 1
#define Y 2
#define DX 3
#define DY 4
#define FX 5
#define FY 6
#define R 7
#define PLANET_SIZE 8
#define G 4.0

__kernel void planetForce(__global float planet[SIZE][PLANET_SIZE])
{
	int i = get_global_id(0);
	for(int j=0;j<SIZE;j++){
		float F,rx,ry;
		rx = planet[i][X]-planet[j][X];
		ry = planet[i][Y]-planet[j][Y];
		F = (planet[i][MASS]*planet[j][MASS]*G)/((rx*rx)+(ry*ry)+0.00000000000001f);
		planet[i][FX] += (-F*rx);
		planet[i][FY] += (-F*ry);
	}
	planet[i][DX] += planet[i][FX]/planet[i][MASS];
	planet[i][DY] += planet[i][FY]/planet[i][MASS];
	planet[i][X]  += planet[i][DX];
	planet[i][Y]  += planet[i][DY];
	planet[i][FX]  = 0;
	planet[i][FY]  = 0;
}