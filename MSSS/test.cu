#include "test.hpp"
#include "thrust/transform_reduce.h"
#include "thrust/functional.h"
#include "thrust/generate.h"
#include "thrust/host_vector.h"
#include "thrust/execution_policy.h"

//template<typename T>
//struct meanStd : public thrust::binary_function<T, T>
//{

//};
void thrustTest()
{
    srand(13);
    size_t N = 10;
    int *h_raw_ptr = (int*)malloc(N * sizeof(int));
    for(int i = 0; i < N; i++)
    {
        h_raw_ptr[i] = i + 1;
    }

    int *raw_ptr;
    gpuErrChk(cudaMalloc((void**)&raw_ptr, N * sizeof(int)));
    gpuErrChk(cudaMemcpy(raw_ptr, h_raw_ptr, N * sizeof(int), cudaMemcpyHostToDevice));
    thrust::device_ptr<int> dev_ptr = thrust::device_pointer_cast(raw_ptr);


    thrust::device_vector<int> dev_vec(raw_ptr, raw_ptr + N);
    int sum = thrust::reduce(thrust::device, dev_vec.begin(), dev_vec.end());
    printf("Sum = %d\n", sum);
    thrust::host_vector<int> host_vec = dev_vec;
    thrust::generate(thrust::host, host_vec.begin(), host_vec.end(), rand);

    for(int i = 0; i < host_vec.size(); i++)
    {
        std::cout << "vec = " << host_vec[i] << std::endl;
    }

    int *host_ptr = (int*)malloc(N * sizeof(int));
    gpuErrChk(cudaMemcpy(host_ptr, raw_ptr, N * sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(raw_ptr);

}

