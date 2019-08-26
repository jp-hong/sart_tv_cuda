#ifndef _PARAMETERS
#define _PARAMETERS

#include <cmath>

#define INPUT_DIR "D:\\Data\\Catphan\\"
#define OUTPUT_DIR "D:\\Data\\SART test\\"


#define SAVE_FILE_NAME "test_cuda75_"
#define SAVE_INTERVAL 10

#define N_ITER 20
#define LAMDA 1.0f
#define RED_REG 0.98f

#define USE_TV true
#define N_TV 20
#define GF 0.2f
#define RED_FACT 0.99f
#define CONV 0.5f
#define MEPS 0.0f

#define PI 3.1415926535f

#define NX 512
#define NY NX
#define NZ 512

#define NS 1440
#define R_NS 60
#define N_SKIP NS / R_NS
#define SLEN PI * 2.f
#define S0 .0f

//#define NU 256
//#define DU 4.0f
//#define ULEN NU * DU
//#define U0 -ULEN / 2.f
//
//#define NV 200
//#define DV 4.0f
//#define VLEN NV * DV
//#define V0 -VLEN / 2.f

#define NU 1024
#define DU 0.398f
#define ULEN NU * DU
#define U0 -ULEN / 2.f

#define NV 1024
#define DV 0.398f
#define VLEN NV * DV
#define V0 -VLEN / 2.f

//#define R 1100.0f
//#define D 1500.0f
//
//#define XLEN 460.f
//#define YLEN 460.f
//#define ZLEN 460.f

#define R 800.0f
#define D 1500.0f

#define XLEN 2.f * R * sinf(atanf(ULEN / 2.f / D))
#define YLEN XLEN
#define ZLEN 2.f * R * sinf(atanf(VLEN / 2.f / D))

#define DX XLEN / NX
#define DY YLEN / NY
#define DZ ZLEN / NZ

#define X0 -XLEN / 2.f
#define Y0 -YLEN / 2.f
#define Z0 -ZLEN / 2.f

#define IMAGE_LEN NX * NY * NZ
#define IMAGE_BYTES IMAGE_LEN * sizeof(float)

#define PROJ_LEN NU * NV
#define PROJ_BYTES PROJ_LEN * sizeof(float)

#endif