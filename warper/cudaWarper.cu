#include "cudaWarper.hpp"
#include "opencv2/core/cuda/saturate_cast.hpp"

namespace cudaMapper {
namespace build_warp_maps {
__constant__ float ck_rinv[9];
__constant__ float cr_kinv[9];
__constant__ float ct[3];
__constant__ float cscale;
}

class SphericalMapper
{
public:
    SphericalMapper() {}
    static __device__ __forceinline__ void mapBackward(float u, float v, float &x, float &y)
    {
        using namespace build_warp_maps;

        v /= cscale;
        u /= cscale;

        float sinv = sinf(v);
        float x_ = sinv * sinf(u);
        float y_ = -cosf(v);
        float z_ = sinv * cosf(u);

        float z;
        x = ck_rinv[0] * x_ + ck_rinv[1] * y_ + ck_rinv[2] * z_;
        y = ck_rinv[3] * x_ + ck_rinv[4] * y_ + ck_rinv[5] * z_;
        z = ck_rinv[6] * x_ + ck_rinv[7] * y_ + ck_rinv[8] * z_;

        if (z > 0) { x /= z; y /= z; }
        else x = y = -1;
    }
};

template <typename Mapper>
__global__ void buildWarpMapsKernel(int tl_u, int tl_v, int cols, int rows,
                                    float * map_x, float * map_y)
{
    int du = blockIdx.x * blockDim.x + threadIdx.x;
    int dv = blockIdx.y * blockDim.y + threadIdx.y;
    if(du < cols && dv < rows)
    {
        float u = tl_u + du;
        float v = tl_v + dv;
        float x, y;
        Mapper::mapBackward(u,v,x,y);
        map_x[dv * cols + du] = x;
        map_y[dv * cols + du] = y;
    }
}

void buildWarpSphericalMaps(int tl_u, int tl_v,
                            float * map_x, float * map_y,
                            int map_rows, int map_cols,
                            const float k_rinv[9], const float r_kinv[9], float scale)
{
    gpuErrChk(cudaMemcpyToSymbol(build_warp_maps::ck_rinv, k_rinv, 9 * sizeof(float)));
    gpuErrChk(cudaMemcpyToSymbol(build_warp_maps::cr_kinv, r_kinv, 9 * sizeof(float)));
    gpuErrChk(cudaMemcpyToSymbol(build_warp_maps::cscale, &scale, sizeof(float)));

    dim3 blockDim(threadsPerBlock, threadsPerBlock);
    dim3 gridDim(ceil(map_cols/threadsPerBlock) + 1, ceil(map_rows/threadsPerBlock) + 1);
    buildWarpMapsKernel<SphericalMapper><<<gridDim, blockDim>>>(tl_u, tl_v, map_cols, map_rows, map_x, map_y);
    cudaDeviceSynchronize();
}
}

