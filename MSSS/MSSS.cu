#include "MSSS.hpp"

__global__ void cudaGray2lab(unsigned char *src, unsigned char *L, unsigned char *a, unsigned char *b, int width, int height)
{
    int row = threadIdx.y + IMUL(blockIdx.y, blockDim.y);
    int col = threadIdx.x + IMUL(blockIdx.x, blockDim.x);

    if(row < height && col < width)
    {
        int idx = row * width + col;
        double data = double(src[idx]) / 255.0;
        double res;
        data = (data > 0.04045) ? powf((data + 0.055)/ 1.055, 2.4):(data / 12.92);
        res = (data > 0.008856f) ? (116 * cbrt(data) - 16): (903.3 * data);
        L[idx] = (res * 2.55);

        a[idx] = 128;
        b[idx] = 128;
    }
}


void MSSS::gpuGray2lab(unsigned char *src, unsigned char *L, unsigned char *a, unsigned char *b, int width, int height)
{
    if(width != m_width || height != m_height)
    {
        LOG_MSG("!!! ERROR: Invalid input size in function %s - file %s - line %d\n", __FUNCTION__, __FILE__, __LINE__);
    }

    dim3 blockDim(threadsPerBlock, threadsPerBlock);
    dim3 gridDim(ceil(double(width)/threadsPerBlock), ceil(double(height)/threadsPerBlock));

    cudaGray2lab<<<gridDim, blockDim>>>(src, L, a, b, width, height);
    gpuErrChk(cudaDeviceSynchronize());
}

__device__ double d_computeIntegralSum(int32_t* integralimg, int x1, int y1, int x2, int y2, int width)
{
    double sum;
    sum = integralimg[IMUL(y2, width) + x2] + integralimg[IMUL(y1 - 1, width) + x1 - 1]
            - integralimg[IMUL(y1 - 1, width) + x2] - integralimg[IMUL(y2, width) + x1 - 1];
//    if(sum < 0)
//    {
//        printf("%d - %d - %d - %d\n", integralimg[IMUL(y2, width) + x2],
//                integralimg[IMUL(y1 - 1, width) + x1 - 1],
//                integralimg[IMUL(y1 - 1, width) + x2],
//                integralimg[IMUL(y2, width) + x1 - 1]);
////        printf("(x1 - y1) - (x2 - y2) = (%d - %d) - (%d - %d)\n", x1,y1,x2,y2);
//    }

    return sum;
}

__global__ void cudaAdd(int32_t * src, double val, int width, int height)
{
    int row = threadIdx.y + IMUL(blockIdx.y, blockDim.y);
    int col = threadIdx.x + IMUL(blockIdx.x, blockDim.x);
    if(row < height && col < width)
    {
        int idx = IMUL(row, width) + col;
        src[idx] = src[idx] + val;
    }
}

void MSSS::gpuAdd(cv::cuda::GpuMat src, double val)
{
    if(src.cols != m_width || src.rows != m_height)
    {
        LOG_MSG("!!! ERROR: Invalid input size in function %s - file %s - line %d\n", __FUNCTION__, __FILE__, __LINE__);
    }
    dim3 blockDim(threadsPerBlock, threadsPerBlock);
    dim3 gridDim(ceil(double(m_width)/threadsPerBlock), ceil(double(m_height)/threadsPerBlock));
    cudaAdd<<<gridDim, blockDim>>>((int32_t*)src.data, val, m_width, m_height);
    gpuErrChk(cudaDeviceSynchronize());
}

__global__ void cudaGetSaliencyMap(unsigned char * L, unsigned char * a, unsigned char *b,
                                   int32_t *inteL, int32_t *intea, int32_t *inteb,
                                   double *saliencyMap, int width, int height)
{
    int row = threadIdx.y + IMUL(blockIdx.y, blockDim.y);
    int col = threadIdx.x + IMUL(blockIdx.x, blockDim.x);
    if(row < height && col < width)
    {
        int idx = IMUL(row, width) + col;
        int y0 = min(row, height - row);
        int y1 = max(0, row - y0);
        int y2 = min(row + y0, height - 1);
        int x0 = min(col, width - col);
        int x1 = max(0, col - x0);
        int x2 = min(col + x0, width - 1);
        double invar = 1.0 /double((y2 - y1 + 1) * (x2 - x1 + 1));
        double sumL = invar * d_computeIntegralSum(inteL, x1 + 1, y1 + 1, x2 + 1, y2 + 1,width + 1);
        double suma = invar * d_computeIntegralSum(intea, x1 + 1, y1 + 1, x2 + 1, y2 + 1,width + 1);
        double sumb = invar * d_computeIntegralSum(inteb, x1 + 1, y1 + 1, x2 + 1, y2 + 1,width + 1);

        saliencyMap[idx] = pow(L[idx] - sumL, 2) +
                pow(a[idx] - suma, 2) +
                pow(b[idx] - sumb, 2);
    }
}

void MSSS::gpuGetSaliencyMap(unsigned char * L, unsigned char * a, unsigned char *b,
                             int32_t *inteL, int32_t *intea, int32_t *inteb,
                             double *saliencyMap)
{

    dim3 blockDim(threadsPerBlock, threadsPerBlock);
    dim3 gridDim(ceil(double(m_width)/threadsPerBlock), ceil(double(m_height)/threadsPerBlock));
    cudaGetSaliencyMap<<<gridDim, blockDim>>>(L, a, b, inteL, intea, inteb, saliencyMap, m_width, m_height);
    gpuErrChk(cudaDeviceSynchronize());
}

__global__ void cudaCreateBinaryMask(double * src, unsigned char*dst, double threshold, int width, int height)
{
    int row = threadIdx.y + IMUL(blockIdx.y, blockDim.y);
    int col = threadIdx.x + IMUL(blockIdx.x, blockDim.x);
    if(row < height && col < width)
    {
        int idx = IMUL(row, width) + col;
        dst[idx] = (double)src[idx] >= threshold ? 255 : 0;
    }
}

void MSSS::gpuCreateBinaryMask(cv::cuda::GpuMat src, cv::cuda::GpuMat dst, double threshold)
{
    dim3 blockDim(threadsPerBlock, threadsPerBlock);
    dim3 gridDim(ceil(double(m_width)/threadsPerBlock), ceil(double(m_height)/threadsPerBlock));
    cudaCreateBinaryMask<<<gridDim, blockDim>>>((double*)src.data, dst.data, threshold, m_width, m_height);
    gpuErrChk(cudaDeviceSynchronize());
}

template<typename T>
__global__ void cudaCopyNonContiguous(T * src, T * dst, size_t step, int width, int height)
{
    int row = threadIdx.y + IMUL(blockIdx.y, blockDim.y);
    int col = threadIdx.x + IMUL(blockIdx.x, blockDim.x);
    if(row < height && col < width)
    {
        int dstIdx = col + IMUL(row, width);
        int srcIdx = IMUL(step, row) + col * sizeof(T);

        dst[dstIdx] = (T)*((unsigned char *)src + srcIdx);
        printf("dst[%d] = %d - src[%d] = %d: srcIdx - row - col = %d - %d - %d\n", dstIdx, dst[dstIdx],srcIdx, (T)*((unsigned char *)src + srcIdx), srcIdx, row, col);
    }
}

template<typename T>
void gpuCopyNonContiguous(T * src, T * dst, size_t step, int width, int height)
{
    dim3 blockDim(threadsPerBlock, threadsPerBlock);
    dim3 gridDim(ceil((float)width/threadsPerBlock), ceil((float)height/threadsPerBlock));
    cudaCopyNonContiguous<T()><<<gridDim, blockDim>>>(src, dst, step, width, height);
    gpuErrChk(cudaDeviceSynchronize());

}
void MSSS::gpuCopyNonContiguous(cv::cuda::GpuMat src, cv::cuda::GpuMat dst, int type)
{
    if(src.cols != dst.cols || src.rows != dst.rows)
    {
        LOG_MSG("!!! ERROR: Src and dst must have the same size at %s - %d \n", __FUNCTION__, __LINE__);
    }

    if(src.type() != dst.type())
    {
        LOG_MSG("!!! ERROR: Src and dst must have the same type at %s - %d \n", __FUNCTION__, __LINE__);
    }

    dim3 blockDim(threadsPerBlock, threadsPerBlock);
    dim3 gridDim(ceil((float)src.cols/threadsPerBlock), ceil((float)src.rows/threadsPerBlock));
    int width = src.cols;
    int height = src.rows;
    std::cout << "step = " << src.step << std::endl;
    switch (type) {
    case 0:
        cudaCopyNonContiguous<unsigned char><<<gridDim, blockDim>>>(src.data, dst.data, src.step, width, height);
        break;
    case 1:
        cudaCopyNonContiguous<int><<<gridDim, blockDim>>>((int*)src.data, (int*)dst.data, src.step, width, height);
        break;
    case 2:
        cudaCopyNonContiguous<int32_t><<<gridDim, blockDim>>>((int32_t*)src.data, (int32_t*)dst.data, src.step, width, height);
        break;
    case 3:
        cudaCopyNonContiguous<float><<<gridDim, blockDim>>>((float*)src.data, (float*)dst.data, src.step, width, height);
        break;
    case 4:
        cudaCopyNonContiguous<double><<<gridDim, blockDim>>>((double*)src.data, (double*)dst.data, src.step, width, height);
        break;
    default:
        break;
    }
    gpuErrChk(cudaDeviceSynchronize());

}
