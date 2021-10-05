#include "utils.hpp"

void printMat(const char *mess, float *src, int width, int height)
{
    float * hSrc = (float*)malloc(width * height * sizeof(float));
    cudaMemcpy(hSrc, src, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    for(int y = 0; y < height; y++)
    {
        for(int x = 0; x < width; x++)
        {
            int idx = y * width + x;

            printf("%.3f\t", hSrc[idx]);

        }
        printf("\n");
    }
    free(hSrc);
}
