#include "../multicue.hpp"
namespace multiCue
{
__global__ void cudaResize(const unsigned char *d_src, unsigned char *d_dst,
                           int srcWidth, int srcHeight,
                           int dstWidth, int dstHeight,
                           double resizeFactorWidth, double resizeFactorHeight)
{
    int row = threadIdx.y + IMUL(blockIdx.y, blockDim.y);
    int col = threadIdx.x + IMUL(blockIdx.x, blockDim.x);

    if(row < dstHeight && col < dstWidth)
    {
        int srcRow = (int)(row * resizeFactorHeight);
        int srcCol = (int)(col * resizeFactorWidth);
        int dstIdx = IMUL(row, dstWidth) + col;
        int srcIdx = IMUL(srcRow, srcWidth) + srcCol;

        d_dst[dstIdx] = d_src[srcIdx];
    }
}

void MultiCues::gpuResize(const unsigned char *d_src, unsigned char *d_dst,
                          int srcWidth, int srcHeight,
                          int dstWidth, int dstHeight)
{
    double resizeFactor_w = (double)srcWidth / (double)dstWidth;
    double resizeFactor_h = (double)srcHeight / (double)dstHeight;
    dim3 blockDim(threadsPerBlock, threadsPerBlock);
    dim3 gridDim(ceil(float(dstWidth)/threadsPerBlock), ceil(float(dstHeight)/threadsPerBlock));

    cudaResize<<<gridDim, blockDim>>>(d_src, d_dst, srcWidth, srcHeight,
                                      dstWidth, dstHeight,
                                      resizeFactor_w, resizeFactor_h);
    gpuErrChk(cudaDeviceSynchronize());
}

__global__ void cudaTextureModelConstruction(short nTrainVolRange, float learningRate,
                                             unsigned char * gray, short2 * neighborPixel,
                                             TextureModel * TModel,
                                             TextureCodeword * TCodeWord,
                                             short* d_aTReferredIndex,
                                             short * d_aTContinuousCnt,
                                             int radius,
                                             bool d_AbsorptionEnable,
                                             int rWidth, int rHeight, int neighborNum, bool * mask)
{
    int tidX = threadIdx.x + IMUL(blockIdx.x, blockDim.x);
    int tidY = threadIdx.y + IMUL(blockIdx.y, blockDim.y);

    int row = tidY;
    int col = tidX / neighborNum;
    int channel = tidX % neighborNum;

    //    if(tidX < rWidth * neighborNum && tidY < rHeight)
    {
        if(mask != nullptr)
        {
            if(mask[IMUL(row, rWidth) + col] != true)
                return;
        }

        if(row < radius || col < radius || row >= rHeight - radius || col >= rWidth - radius)
            return;

        int srcIdx = IMUL(row, rWidth) + col;
        int idx = IMUL(srcIdx, neighborNum) + channel;

        short2 *aNei = neighborPixel + idx;
        TextureModel * aModel = TModel + idx;
        TextureCodeword * aCodeWord = TCodeWord + aModel->m_CodewordsIdx;

        // for all neighboring pair
        float negLearningRate = 1 - learningRate;
        float fDiff;

        fDiff = (float)(gray[srcIdx] - gray[IMUL(aNei->y, rWidth) + aNei->x]);

        // Step 1: Matching

        int iMatchedIndex = -1;

        for(int j = 0; j < aModel->m_iNumEntries; j++)
        {
            if(aCodeWord[j].m_fLowThre <= fDiff && fDiff <= aCodeWord[j].m_fHighThre)
            {
                iMatchedIndex = j;
                break;
            }
        }

        aModel->m_iTotal++;

        // Step 2: adding a new element
        if(iMatchedIndex == -1)
        {
            // element array
            if(aModel->m_iElementArraySize == aModel->m_iNumEntries)
            {
                aModel->m_iElementArraySize += 5;
                assert(aModel->m_iElementArraySize <= CUDA_MAX_CODEWORDS_SIZE);
            }

            aCodeWord[aModel->m_iNumEntries].m_fMean = fDiff;
            aCodeWord[aModel->m_iNumEntries].m_fLowThre = aCodeWord[aModel->m_iNumEntries].m_fMean - nTrainVolRange;
            aCodeWord[aModel->m_iNumEntries].m_fHighThre = aCodeWord[aModel->m_iNumEntries].m_fMean + nTrainVolRange;
            aCodeWord[aModel->m_iNumEntries].m_iT_first_time = aModel->m_iTotal;
            aCodeWord[aModel->m_iNumEntries].m_iT_last_time  = aModel->m_iTotal;
            aCodeWord[aModel->m_iNumEntries].m_iMNRL = aModel->m_iTotal - 1;
            aModel->m_iNumEntries++;
        }

        // Step 3: Update
        else
        {
            float fDiffMean = aCodeWord[iMatchedIndex].m_fMean;
            aCodeWord[iMatchedIndex].m_fMean = learningRate * fDiff + negLearningRate * fDiffMean;
            aCodeWord[iMatchedIndex].m_fLowThre = aCodeWord[iMatchedIndex].m_fMean - nTrainVolRange;
            aCodeWord[iMatchedIndex].m_fHighThre = aCodeWord[iMatchedIndex].m_fMean + nTrainVolRange;
            aCodeWord[iMatchedIndex].m_iT_last_time = aModel->m_iTotal;
        }

        // cache-book handling
        if(aModel->m_bID == 1)
        {
            // 1. m_iMNRL update
            int negTime;
            for(int j = 0; j < aModel->m_iNumEntries;j++)
            {
                negTime = aModel->m_iTotal - aCodeWord[j].m_iT_last_time + aCodeWord[j].m_iT_first_time - 1;
                if(aCodeWord[j].m_iMNRL < negTime)
                    aCodeWord[j].m_iMNRL = negTime;
            }

            // 2. TReferredIndex update
            if(d_AbsorptionEnable == true)
            {
                d_aTReferredIndex[idx] = -1;
            }
        }
        else
        {
            // 1. m_iMNRL Update
            if(iMatchedIndex == -1)
            {
                aCodeWord[aModel->m_iNumEntries - 1].m_iMNRL = 0;
            }

            // 2. TReferredIndex update
            if(iMatchedIndex == -1)
            {
                d_aTReferredIndex[idx] = aModel->m_iNumEntries - 1;
                d_aTContinuousCnt[idx] = 1;
            }
            else
            {
                if(iMatchedIndex == d_aTReferredIndex[idx])
                    d_aTContinuousCnt[idx]++;
                else
                {
                    d_aTReferredIndex[idx] = iMatchedIndex;
                    d_aTContinuousCnt[idx] = 1;
                }
            }
        }
    }
}

void MultiCues::gpuTextureModelConstruction(short nTrainVolRange, float learningRate,
                                            unsigned char *gray, short2 * neighborDirection,
                                            TextureModel *TModel,
                                            TextureCodeword * TCodeWord,
                                            short * TReferredIndex,
                                            short * TContinuousCnt,
                                            int radius, int rWidth, int rHeight,
                                            int neighborNum, bool * mask)
{
    dim3 blockDim(threadsPerBlock, threadsPerBlock);
    dim3 gridDim(ceil((float)rWidth * neighborNum / threadsPerBlock), ceil((float)rHeight / threadsPerBlock));

    cudaTextureModelConstruction<<<gridDim, blockDim>>>(nTrainVolRange, learningRate, gray, neighborDirection, TModel,
                                                        TCodeWord, TReferredIndex, TContinuousCnt,
                                                        radius, m_AbsorptionEnable, rWidth, rHeight, neighborNum, mask);
    gpuErrChk(cudaDeviceSynchronize());
}

__global__ void cudaColorCodeBookConstruction(short nTrainVolRange, float learningRate,
                                              unsigned char *gray,
                                              ColorModel *CModel,
                                              ColorCodeword *CCodeword,
                                              short *CReferredIndex,
                                              short *CContinuousCnt,
                                              int radius,
                                              bool absorptionEnable,
                                              int rWidth, int rHeight, bool *mask)
{
    int row = threadIdx.y + IMUL(blockIdx.y, blockDim.y);
    int col = threadIdx.x + IMUL(blockIdx.x, blockDim.x);

    if(row < rHeight && col < rWidth)
    {
        int idx = col + IMUL(row, rWidth);
        if(mask != nullptr)
        {
            if(mask[idx] == false)
                return;
        }
        //        printf("height - width = %d - %d\n", rHeight, rWidth);
        if(col < radius || row < radius ||
                col >= rWidth - radius || row >= rHeight - radius)
        {
            return;
        }

        ColorModel *pC = CModel + idx;
        // step 1: matching
        short nMatchedIndex;
        float negLearningRate = 1 - learningRate;

        nMatchedIndex = -1;

        for(int i = 0; i < pC->m_iNumEntries; i++)
        {
            if(CCodeword[pC->m_CodewordsIdx + i].m_dMean - nTrainVolRange <= gray[idx] &&
                    gray[idx] <= CCodeword[pC->m_CodewordsIdx + i].m_dMean + nTrainVolRange)
            {
                nMatchedIndex = i;
                break;
            }
        }
        //        printf("row - col - nMatchedIndex = %d - %d - %d\n", row, col, nMatchedIndex);
        pC->m_iTotal++;
        // Step 2: Add a new element
        if(nMatchedIndex == -1)
        {
            if (pC->m_iElementArraySize == pC->m_iNumEntries) {
                pC->m_iElementArraySize = pC->m_iElementArraySize + 5;
                assert(pC->m_iElementArraySize <= CUDA_MAX_CODEWORDS_SIZE);
            }

            CCodeword[pC->m_CodewordsIdx + pC->m_iNumEntries].m_dMean = gray[idx];
            CCodeword[pC->m_CodewordsIdx + pC->m_iNumEntries].m_iT_first_time = pC->m_iTotal;
            CCodeword[pC->m_CodewordsIdx + pC->m_iNumEntries].m_iT_last_time = pC->m_iTotal;
            CCodeword[pC->m_CodewordsIdx + pC->m_iNumEntries].m_iMNRL = pC->m_iTotal - 1;
            pC->m_iNumEntries = pC->m_iNumEntries + 1;
        }
        // Step3: Update
        else {
            CCodeword[pC->m_CodewordsIdx + nMatchedIndex].m_dMean = (learningRate * gray[idx]) +
                    negLearningRate * CCodeword[pC->m_CodewordsIdx + nMatchedIndex].m_dMean;
            CCodeword[pC->m_CodewordsIdx + nMatchedIndex].m_iT_last_time = pC->m_iTotal;
        }

        // Cachebook Handling
        if (pC->m_bID == 1) {
            //1. m_iMNRL update
            int iNegTime;
            for (int i = 0; i < pC->m_iNumEntries; i++) {
                //m_iMNRL update
                iNegTime = pC->m_iTotal - CCodeword[pC->m_CodewordsIdx + i].m_iT_last_time
                        + CCodeword[pC->m_CodewordsIdx + i].m_iT_first_time - 1;
                if (CCodeword[pC->m_CodewordsIdx + i].m_iMNRL < iNegTime) CCodeword[pC->m_CodewordsIdx + i].m_iMNRL = iNegTime;
            }

            //2. g_aCReferredIndex[iPosY][iPosX] update
            if (absorptionEnable == true)
                CReferredIndex[idx] = -1;
        }

        else {
            //1. m_iMNRL update:
            if (nMatchedIndex == -1)
                CCodeword[pC->m_CodewordsIdx + pC->m_iNumEntries - 1].m_iMNRL = 0;

            //2. g_aCReferredIndex[iPosY][iPosX] update
            if (nMatchedIndex == -1) {
                CReferredIndex[idx] = pC->m_iNumEntries - 1;
                CContinuousCnt[idx] = 1;
            }
            else {
                if (nMatchedIndex == CReferredIndex[idx])
                    CContinuousCnt[idx]++;
                else {
                    CReferredIndex[idx] = nMatchedIndex;
                    CContinuousCnt[idx] = 1;
                }
            }
        }
    }
}

void MultiCues::gpuColorCodeBookConstruction(short nTrainVolRange, float learningRate,
                                             unsigned char *gray,
                                             ColorModel *CModel,
                                             ColorCodeword *CCodeword,
                                             short *CReferredIndex,
                                             short *CContinuousCnt,
                                             int radius,
                                             int rWidth, int rHeight, bool *mask)
{
    dim3 blockDim(threadsPerBlock, threadsPerBlock);
    dim3 gridDim(ceil((float)rWidth / threadsPerBlock), ceil((float)rHeight / threadsPerBlock));

    cudaColorCodeBookConstruction<<<gridDim, blockDim>>>(nTrainVolRange, learningRate, gray,
                                                         CModel,CCodeword, CReferredIndex, CContinuousCnt,
                                                         radius, m_AbsorptionEnable, rWidth, rHeight, mask);
    gpuErrChk(cudaDeviceSynchronize());

}
__global__ void cudaSetNeighborDirection(short2* neigborDirection, short2 * d_SearchDirection, int rWidth, int rHeight, int neighborNum)
{
    int tidX = threadIdx.x + IMUL(blockIdx.x, blockDim.x);
    int tidY = threadIdx.y + IMUL(blockIdx.y, blockDim.y);

    int row = tidY;
    int col = tidX / neighborNum;
    int channel = tidX % neighborNum;


    if(tidX < rWidth * neighborNum && tidY < rHeight)
    {
        int idx = (row * rWidth + col) * neighborNum + channel;
        short2 temp_pos;
        temp_pos.x = col + d_SearchDirection[channel].x;
        temp_pos.y = row + d_SearchDirection[channel].y;
        if(temp_pos.x < 0 || temp_pos.x >= rWidth || temp_pos.y < 0 || temp_pos.y >= rHeight)
        {
            neigborDirection[idx].x = -1;
            neigborDirection[idx].y = -1;
        }
        else
        {
            neigborDirection[idx].x = temp_pos.x;
            neigborDirection[idx].y = temp_pos.y;
        }
    }
}

void MultiCues::setNeighborDirection()
{
    short2 * h_SearchDirection;
    gpuErrChk(cudaMallocHost((void**)&h_SearchDirection, sizeof(short2) * m_neighborNum));
    h_SearchDirection[0].x = -2;
    h_SearchDirection[0].y = 0;     // 180 degree

    h_SearchDirection[1].x = -1;    //135 degree
    h_SearchDirection[1].y = -2;

    h_SearchDirection[2].x = 1;     //45 degree
    h_SearchDirection[2].y = -2;

    h_SearchDirection[3].x = 2;     //0 degree
    h_SearchDirection[3].y = 0;

    h_SearchDirection[4].x = 1;     //-45 degree
    h_SearchDirection[4].y = 2;

    h_SearchDirection[5].x = -1;    //-135 degree
    h_SearchDirection[5].y = 2;

    short2* d_SearchDirection;
    gpuErrChk(cudaMalloc((void**)&d_SearchDirection, sizeof(short2) * m_neighborNum));
    gpuErrChk(cudaMemcpy(d_SearchDirection, h_SearchDirection, sizeof(short2) * m_neighborNum, cudaMemcpyHostToDevice));

    gpuErrChk(cudaFreeHost(h_SearchDirection));

    dim3 blockDim(threadsPerBlock, threadsPerBlock);
    dim3 gridDim(ceil((float)m_ResizeWidth * m_neighborNum/ threadsPerBlock), ceil((float)m_ResizeHeight/threadsPerBlock));

    cudaSetNeighborDirection<<<gridDim, blockDim>>>(m_neighborDirection, d_SearchDirection,m_ResizeWidth, m_ResizeHeight, m_neighborNum);
    gpuErrChk(cudaDeviceSynchronize());
    gpuErrChk(cudaFree(d_SearchDirection));
}
__global__ void cudaAllocateTextureModelRelateMemoryHelper(TextureModel * TModel, int iElementArraySize, int bID, int maxNum)
{
    int idx = threadIdx.x + IMUL(blockIdx.x, blockDim.x);
    //    printf("idx = %d\n", idx);
    if(idx < maxNum)
    {
        TModel[idx].m_CodewordsIdx = idx * CUDA_MAX_CODEWORDS_SIZE;
        TModel[idx].m_iElementArraySize = iElementArraySize;
        TModel[idx].m_iNumEntries = 0;
        TModel[idx].m_iTotal = 0;
        TModel[idx].m_bID = bID;
    }
}

void MultiCues::allocateTextureModelRelatedMemoryHelper(TextureModel *TModel, int iElementArraySize, int _bID,
                                                        int rWidth, int rHeight, int neighborNum)
{
    cudaAllocateTextureModelRelateMemoryHelper<<<(rWidth * rHeight * neighborNum + threadsPerBlock - 1) / threadsPerBlock + 1, threadsPerBlock>>>(TModel, iElementArraySize, _bID,
                                                                                                                                                  rWidth * rHeight * neighborNum);
    gpuErrChk(cudaDeviceSynchronize());
}

__global__ void cudaAllocateColorModelRelateMemoryHelper(ColorModel * CModel, int iElementArraySize, int _bID, int maxNum)
{
    int idx = threadIdx.x + IMUL(blockIdx.x, blockDim.x);
    //    printf("idx = %d\n", idx);
    if(idx < maxNum)
    {
        CModel[idx].m_CodewordsIdx = idx * CUDA_MAX_CODEWORDS_SIZE;
        CModel[idx].m_iNumEntries = 0;
        CModel[idx].m_iElementArraySize = iElementArraySize;
        CModel[idx].m_iTotal = 0;
        CModel[idx].m_bID = _bID;
    }
}

void MultiCues::allocateColorModelRelatedMemoryHelper(ColorModel *CModel, int iElementArraySize, int _bID, int rWidth, int rHeight)
{
    cudaAllocateColorModelRelateMemoryHelper<<<(rWidth * rHeight + threadsPerBlock - 1) / threadsPerBlock + 1, threadsPerBlock>>>(CModel, iElementArraySize, _bID,
                                                                                                                                  rWidth * rHeight);
    gpuErrChk(cudaDeviceSynchronize());
}

__global__ void cudaTextureClearNonEssentialEntries(short nClearNum, TextureModel * TModel,
                                                    TextureCodeword * TCodeword, int * keepCnt,
                                                    TextureCodeword * TCodewordTemp, bool *mask,
                                                    int rWidth, int rHeight, int neighborNum)
{
    int tidX = threadIdx.x + IMUL(blockIdx.x, blockDim.x);
    int tidY = threadIdx.y + IMUL(blockIdx.y, blockDim.y);

    int row = tidY;
    int col = tidX / neighborNum;
    int channel = tidX % neighborNum;
    if(tidX < rWidth * neighborNum && tidY < rHeight)
    {
        int srcIdx = IMUL(row, rWidth) + col;
        if(mask != nullptr)
        {
            if(mask[srcIdx] == false)
                return;
        }
        int idx = IMUL(neighborNum, srcIdx) + channel;

        TextureModel *aModel = TModel + idx;
        TextureCodeword * aCodeword = TCodeword + aModel->m_CodewordsIdx;
        int * aKeep =  keepCnt + idx * CUDA_MAX_CODEWORDS_SIZE;

        if(aModel->m_iTotal < nClearNum)
        {
            return;
        }

        int i;
        int iStaleThresh = (int) nClearNum * 0.5;
        int iKeepCnt = 0;
        // Step 2: Find non-essential codewords
        for(i = 0; i < aModel->m_iNumEntries; i++)
        {
            if(aCodeword[i].m_iMNRL > iStaleThresh)
            {
                aKeep[i] = 0;
            }
            else
            {
                aKeep[i] = 1;
                iKeepCnt++;
            }
        }
        //Step3: Perform removal
        if (iKeepCnt == 0 || iKeepCnt == aModel->m_iNumEntries) {
            for (i = 0; i < aModel->m_iNumEntries; i++) {
                aCodeword[i].m_iT_first_time = 1;
                aCodeword[i].m_iT_last_time = 1;
                aCodeword[i].m_iMNRL = 0;
            }
        }
        else {
            iKeepCnt = 0;
            TextureCodeword * temp = TCodewordTemp + idx * CUDA_MAX_CODEWORDS_SIZE;
            for (i = 0; i < aModel->m_iNumEntries; i++) {
                if (aKeep[i] == 1) {
                    temp[iKeepCnt] = aCodeword[i];
                    temp[iKeepCnt].m_iT_first_time = 1;
                    temp[iKeepCnt].m_iT_last_time = 1;
                    temp[iKeepCnt].m_iMNRL = 0;
                    iKeepCnt++;
                }
            }
            for(i = 0; i < iKeepCnt; i++) {
                aCodeword[i] = temp[i];
            }

            aModel->m_iElementArraySize = aModel->m_iNumEntries;
            aModel->m_iNumEntries = iKeepCnt;
        }
        aModel->m_iTotal = 0;
    }
}

void MultiCues::gpuTextureClearNonEssentialEntries(short nClearNum, TextureModel *aModel, TextureCodeword *aCodeword, bool *mask)
{
    dim3 blockDim(threadsPerBlock, threadsPerBlock);
    dim3 gridDim(ceil((float)m_ResizeWidth * m_neighborNum/threadsPerBlock), ceil((float)m_ResizeHeight/threadsPerBlock));

    struct timeval t1, t2;
    gettimeofday(&t1, 0);
    cudaTextureClearNonEssentialEntries<<<gridDim, blockDim>>>(nClearNum, aModel, aCodeword,
                                                               m_TkeepCnt, m_TCodewordTemp, mask,
                                                               m_ResizeWidth, m_ResizeHeight,
                                                               m_neighborNum);
    gettimeofday(&t2, 0);
    getKernelTime("Clear Kernel Time = %.4f\n", t2, t1);

    gpuErrChk(cudaDeviceSynchronize());
}

__global__ void cudaColorClearNonEssentialEntries(short nClearNum, ColorModel * CModel, ColorCodeword * CCodeword,
                                                  int * keepCnt, ColorCodeword * CCodewordTemp,
                                                  bool *mask, int rWidth, int rHeight)
{
    int row = threadIdx.y + IMUL(blockIdx.y, blockDim.y);
    int col = threadIdx.x + IMUL(blockIdx.x, blockDim.x);

    if(row < rHeight && col < rWidth)
    {
        int idx = col + IMUL(row, rWidth);
        if(mask != nullptr)
        {
            if(mask[idx] == false)
                return;
        }

        ColorModel * aModel = CModel + idx;
        int i;
        short nStaleThresh = (int)(nClearNum * 0.5);
        short nKeepCnt;

        if(aModel->m_iTotal < nClearNum)
            return;

        int * aKeep = keepCnt + idx * CUDA_MAX_CODEWORDS_SIZE;
        //Step2: Find non-essential codewords
        for (i = 0; i < aModel->m_iNumEntries; i++) {
            if (CCodeword[aModel->m_CodewordsIdx + i].m_iMNRL > nStaleThresh) {
                aKeep[i] = 0; //removal
            }
            else {
                aKeep[i] = 1; //keep
                nKeepCnt++;
            }
        }
        //Step3: Perform removal
        if (nKeepCnt == 0 || nKeepCnt == aModel->m_iNumEntries) {
            for (i = 0; i < aModel->m_iNumEntries; i++) {
                CCodeword[aModel->m_CodewordsIdx + i].m_iT_first_time = 1;
                CCodeword[aModel->m_CodewordsIdx + i].m_iT_last_time = 1;
                CCodeword[aModel->m_CodewordsIdx + i].m_iMNRL = 0;
            }
        }
        else {
            nKeepCnt = 0;
            ColorCodeword * temp = CCodewordTemp + idx * CUDA_MAX_CODEWORDS_SIZE;

            for (i = 0; i < aModel->m_iNumEntries; i++) {
                if (aKeep[i] == 1) {
                    temp[nKeepCnt] = CCodeword[aModel->m_CodewordsIdx + i];
                    temp[nKeepCnt].m_iT_first_time = 1;
                    temp[nKeepCnt].m_iT_last_time = 1;
                    temp[nKeepCnt].m_iMNRL = 0;
                    nKeepCnt++;
                }
            }

            for(i = 0; i < nKeepCnt; i++)
                CCodeword[aModel->m_CodewordsIdx + i] = temp[i];
            //ending..
            aModel->m_iElementArraySize = aModel->m_iNumEntries;
            aModel->m_iNumEntries = nKeepCnt;
        }

        aModel->m_iTotal = 0;
    }
}

void MultiCues::gpuColorClearNonEssentialEntries(short nClearNum, ColorModel *aModel, ColorCodeword *aCodeword, bool *mask)
{
    dim3 blockDim(threadsPerBlock, threadsPerBlock);
    dim3 gridDim(ceil((float)m_ResizeWidth/threadsPerBlock), ceil((float)m_ResizeHeight/threadsPerBlock));


    cudaColorClearNonEssentialEntries<<<gridDim, blockDim>>>(nClearNum, aModel, aCodeword, m_CkeepCnt,
                                                             m_CCodewordTemp, mask, m_ResizeWidth, m_ResizeHeight);

    gpuErrChk(cudaDeviceSynchronize());

}

__global__ void cudaGetConfidenceMap(unsigned char *gray, float *aTextureMap,
                                     short2 *neighborDir, TextureModel *TModel,
                                     TextureCodeword *TCodeword, int radius, int rWidth, int rHeight, int neighborNum, float padding)
{
    int row = threadIdx.y + IMUL(blockIdx.y, blockDim.y);
    int col = threadIdx.x + IMUL(blockIdx.x, blockDim.x);

    if(row < rHeight && col < rWidth)
    {
        int idx = col + IMUL(row, rWidth);
        if(row < radius ||row >= rHeight - radius || col < radius || col >= rWidth - radius)
        {
            aTextureMap[idx] = 0;
        }
        else
        {

            int nMatchedCount = 0;
            float fDiff;
            short2 nei;

            for(int i = 0; i < neighborNum; i++)
            {
                int tempIdx = idx * neighborNum + i;
                nei.x = neighborDir[tempIdx].x;
                nei.y = neighborDir[tempIdx].y;

                fDiff = (float)(gray[idx] - gray[nei.x + IMUL(nei.y, rWidth)]);
                int temp_codeword_idx = (TModel + tempIdx)->m_CodewordsIdx;
                for(int j = 0; j < (TModel + tempIdx)->m_iNumEntries; j++)
                {
                    TextureCodeword tempCodeword = TCodeword[temp_codeword_idx + j];
                    if(tempCodeword.m_fLowThre - padding <= fDiff &&
                            fDiff <= tempCodeword.m_fHighThre + padding)
                    {
                        nMatchedCount++;
                        break;
                    }
                }
            }

            aTextureMap[idx] = 1 - (float)nMatchedCount / neighborNum;
        }
    }
}

void MultiCues::gpuGetConfidenceMap(unsigned char *gray, float *aTextureMap,
                                    short2 *neighborDir, TextureModel *TModel,
                                    TextureCodeword *TCodeword)
{
    dim3 blockDim(threadsPerBlock, threadsPerBlock);
    dim3 gridDim(ceil((float)m_ResizeWidth/threadsPerBlock), ceil((float)m_ResizeHeight/threadsPerBlock));
    float padding = 5;

    cudaGetConfidenceMap<<<gridDim, blockDim>>>(gray, aTextureMap, neighborDir, TModel, TCodeword, m_neighborRadius, m_ResizeWidth,
                                                m_ResizeHeight, m_neighborNum, padding);

    gpuErrChk(cudaDeviceSynchronize());
}

__global__ void cudaCreateLandMarkArray(float fConfThresh, short nTrainVolRange, float *aConfMap,
                                        unsigned char *gray, short2 *neighborDir,
                                        TextureModel *TModel, ColorModel *CModel,
                                        TextureCodeword *TCodeword, ColorCodeword *CCodeword,
                                        unsigned char *landMarkMap, int radius,
                                        int rWidth, int rHeight, int neighborNum)
{
    int row = threadIdx.y + IMUL(blockIdx.y, blockDim.y);
    int col = threadIdx.x + IMUL(blockIdx.x, blockDim.x);

    if(row < rHeight && col < rWidth)
    {
        int idx = col + IMUL(row, rWidth);
        if(row < radius ||row >= rHeight - radius || col < radius || col >= rWidth - radius)
        {
            landMarkMap[idx] = 0;
            return;
        }
        float tmp = aConfMap[idx];
        if (tmp > fConfThresh) {
            landMarkMap[idx] = 255;
        }
        else {
            landMarkMap[idx] = 0;
            //Calculating texture amount in the background
            double dBackAmt, dCnt;
            dBackAmt = dCnt = 0;
            for (int m = 0; m < neighborNum; m++) {
                int temp_codewords_idx = (TModel + idx* neighborNum + m)->m_CodewordsIdx;
                for (int n = 0; n < (TModel + idx*neighborNum + m)->m_iNumEntries; n++) {
                    TextureCodeword temp_codewords = TCodeword[temp_codewords_idx + n];
                    dBackAmt += temp_codewords.m_fMean;
                    dCnt++;
                }
            }
            dBackAmt /= dCnt;

            //Calculating texture amount in the input image
            double dTemp, dInputAmt = 0;
            for (int m = 0; m < neighborNum; m++) {
                int temp_idx = idx*neighborNum + m;
                dTemp = gray[idx] - gray[neighborDir[temp_idx].y*rWidth + neighborDir[temp_idx].x];

                if (dTemp >= 0) dInputAmt += dTemp;
                else dInputAmt -= dTemp;
            }

            //If there are only few textures in both background and input image
            if (dBackAmt < 50 && dInputAmt < 50) { // 50
                //Conduct color codebook matching
                bool bMatched = false;
                for (int m = 0; m < CModel[idx].m_iNumEntries; m++) {
                    int temp_index = CModel[idx].m_CodewordsIdx;
                    double dLowThre  = CCodeword[temp_index + m].m_dMean - nTrainVolRange - 15;
                    double dHighThre = CCodeword[temp_index + m].m_dMean + nTrainVolRange + 15;

                    if (dLowThre <= gray[idx] && gray[idx]<= dHighThre) {
                        bMatched = true;
                        break;
                    }

                }
                if (bMatched == true)
                    landMarkMap[idx] = 0;
                else
                    landMarkMap[idx] = 255;

            }

        }
    }
}

void MultiCues::gpuCreateLandMarkArray(float fConfThresh, short nTrainVolRange, float *aConfMap,
                                       unsigned char *gray, short2 *neighborDir,
                                       TextureModel *TModel, ColorModel *CModel,
                                       TextureCodeword *TCodeword, ColorCodeword *CCodeword,
                                       unsigned char *landMarkMap)
{
    dim3 blockDim(threadsPerBlock, threadsPerBlock);
    dim3 gridDim(ceil((float)m_ResizeWidth/threadsPerBlock), ceil((float)m_ResizeHeight/threadsPerBlock));

    cudaCreateLandMarkArray<<<gridDim, blockDim>>>(fConfThresh, nTrainVolRange, aConfMap, gray, neighborDir,
                                                   TModel, CModel, TCodeword, CCodeword, landMarkMap,
                                                   m_neighborRadius, m_ResizeWidth, m_ResizeHeight, m_neighborNum);
    gpuErrChk(cudaDeviceSynchronize());
}

__global__ void cudaUpdateModel(BoundingBoxInfo * bboxInfo,unsigned char * d_ResizeFgMap, bool * updateMap, bool *updateMapCache, int rWidth)
{
    int k = threadIdx.x + IMUL(blockIdx.x, blockDim.x);
    if(k < bboxInfo->boxNum)
    {
        if(bboxInfo->d_isValidBox[k] == true)
        {
            for(int i = bboxInfo->dRbox[k].z; i <= bboxInfo->dRbox[k].w; i++)
            {
                for(int j = bboxInfo->dRbox[k].x; j<= bboxInfo->dRbox[k].y; j++)
                {
                    int idx = j + IMUL(i, rWidth);
                    if(d_ResizeFgMap[idx] == 255)
                    {
                        updateMap[idx] = false;
                        updateMapCache[idx] = true;
                    }
                }
            }
        }
    }
}

void MultiCues::gpuUpdateModel(BoundingBoxInfo *bboxInfo, unsigned char * d_ResizeFgMap, bool *updateMap, bool *updateMapCache)
{
    if(bboxInfo->boxNum > 0)
    {
        cudaUpdateModel<<<(bboxInfo->boxNum - 1)/ threadsPerBlock + 1, threadsPerBlock>>>(bboxInfo, d_ResizeFgMap, updateMap, updateMapCache, m_ResizeWidth);
        gpuErrChk(cudaDeviceSynchronize());
    }
}

__global__ void cudaMorphologicalOperation(const unsigned char * __restrict__ src, unsigned char * __restrict__ dst, int threshold, int offset, int rWidth, int rHeight)
{
    int row = threadIdx.y + IMUL(blockIdx.y, blockDim.y);
    int col = threadIdx.x + IMUL(blockIdx.x, blockDim.x);

    if(row < rHeight && col < rWidth)
    {
        int idx = col + IMUL(row, rWidth);
        if(row < offset || col < offset || row >= rHeight - offset || col >= rWidth - offset)
        {
            dst[idx] = 0;
        }
        else
        {
            extern __shared__ unsigned char sdata[];
            int sharedMemWidth = threadsPerBlock + 2 * offset + 1;
            sdata[IMUL(threadIdx.y + offset, sharedMemWidth) + threadIdx.x + offset] = src[idx];
            if(threadIdx.x < offset && threadIdx.y >= offset)
            {
                sdata[IMUL(threadIdx.y + offset, sharedMemWidth) + threadIdx.x] = src[col - offset + IMUL(row, rWidth)];
                sdata[IMUL(threadIdx.y + offset, sharedMemWidth) + threadIdx.x + offset + threadsPerBlock] = src[col + threadsPerBlock + IMUL(row, rWidth)];
            }

            if(threadIdx.y < offset && threadIdx.x >= offset)
            {
                sdata[IMUL(threadIdx.y, sharedMemWidth) + threadIdx.x + offset] = src[col + IMUL(row - offset, rWidth)];
                sdata[IMUL(threadIdx.y + offset + threadsPerBlock, sharedMemWidth) + threadIdx.x + offset] = src[col + IMUL(row + threadsPerBlock, rWidth)];
            }

            if(threadIdx.x < offset && threadIdx.y < offset)
            {
                sdata[IMUL(threadIdx.y, sharedMemWidth) + threadIdx.x] = src[col - offset + IMUL(row - offset, rWidth)];
                sdata[IMUL(threadIdx.y + offset + threadsPerBlock, sharedMemWidth) + threadIdx.x] = src[col - offset + IMUL(row + threadsPerBlock, rWidth)];
                sdata[IMUL(threadIdx.y, sharedMemWidth) + threadIdx.x + offset + threadsPerBlock] = src[col + threadsPerBlock + IMUL(row - offset, rWidth)];
            }

            int cnt = 0;
            for(int m = -offset; m <= offset; m++)
            {
                for(int n = -offset; n <= offset; n++)
                {
                    if(sdata[IMUL(threadIdx.y + offset + m, sharedMemWidth) + threadIdx.x + offset + n] == 255)
                    {
                        cnt++;
                    }
                }
            }
            if(cnt >= threshold)
                dst[idx] = 255;
            else
                dst[idx] = 0;
        }
    }
}

void MultiCues::gpuMorphologicalOperation(unsigned char *src, unsigned char *dst, float thresholdRatio, int maskSize, int rWidth, int rHeight)
{
    int offset = (int)(maskSize / 2);
    dim3 blockDim(threadsPerBlock, threadsPerBlock);
    dim3 gridDim(ceil((float)rWidth/threadsPerBlock), ceil((float)rHeight/threadsPerBlock));
    int sharedMem = (threadsPerBlock + 2 * offset) * (threadsPerBlock + 2 * offset + 1) * sizeof(unsigned char);

    cudaMorphologicalOperation<<<gridDim, blockDim, sharedMem>>>(src, dst, (int)(thresholdRatio * maskSize * maskSize), offset, rWidth, rHeight);
    gpuErrChk(cudaDeviceSynchronize());

}

__global__ void cudaSetValueByIndex(int * src, int valueNum)
{
    int idx = threadIdx.x + IMUL(blockIdx.x, blockDim.x);
    if(idx < valueNum)
    {
        src[idx] = idx;
    }
}

void gpuSetValueByIndex(int *src, int valueNum)
{
    cudaSetValueByIndex<<<(valueNum - threadsPerBlock + 1)/threadsPerBlock + 1, threadsPerBlock>>>(src, valueNum);
    gpuErrChk(cudaDeviceSynchronize());
}

void MultiCues::gpuLabelling(unsigned char *src, int *labelCnt, int *labelTable)
{
    int * d_pass1;
    int * d_table1;
    int * d_table2;
    int rImgSize = m_ResizeHeight * m_ResizeWidth;
    gpuErrChk(cudaMalloc((void**)&d_pass1, rImgSize * sizeof(int)));
    gpuErrChk(cudaMalloc((void**)&d_table1, rImgSize / 2 * sizeof(int)));
    gpuErrChk(cudaMalloc((void**)&d_table2, rImgSize / 2 * sizeof(int)));

    gpuErrChk(cudaMemset(d_pass1, 0, rImgSize * sizeof(int)));
    gpuErrChk(cudaMemset(labelTable, 0, rImgSize * sizeof(int)));
    gpuSetValueByIndex(d_table1, rImgSize / 2);
    gpuErrChk(cudaMemset(d_table2, 0, rImgSize / 2 * sizeof(int)));
}

__global__ void cudaTextureAbsorption(int absorpCnt, short *TContinuousCnt,
                                      short *TReferredIndex, TextureModel *TModel,
                                      TextureCodeword *TCodeword,
                                      TextureModel *TCacheBook,
                                      TextureCodeword *TCodewordCacheBook,
                                      TextureCodeword *TTempCache, bool * mask,
                                      int neighborNum, int radius,
                                      int rWidth, int rHeight)
{
    int row = threadIdx.y + IMUL(blockIdx.y, blockDim.y);
    int col = threadIdx.x + IMUL(blockIdx.x, blockDim.x);

    if(row < rHeight && col < rWidth)
    {
        int idx = col + IMUL(row, rWidth);
        if(mask != nullptr)
        {
            if(mask[idx] == false)
                return;
        }
        if(row < radius || col < radius || row >= rHeight - radius || col >= rWidth - radius)
            return;


        TextureModel * pModel = TModel + IMUL(idx, neighborNum);
        TextureModel * pCache = TCacheBook + IMUL(idx, neighborNum);

        int i, j, k;
        int leavingIdx;
        for (i = 0; i < neighborNum; i++) {
            //set iLeavingIndex
            if (TContinuousCnt[i + IMUL(idx, neighborNum)] < absorpCnt)
                continue;

            leavingIdx = TReferredIndex[i + IMUL(idx, neighborNum)];
            // no array expansion
            if ((pModel + i)->m_iElementArraySize == (pModel + i)->m_iNumEntries) {
                (pModel + i)->m_iElementArraySize = (pModel + i)->m_iElementArraySize + 5;
                assert((pModel + i)->m_iElementArraySize <= CUDA_MAX_CODEWORDS_SIZE);
            }

            //movement from the cache-book to the codebook
            int top_index = (pModel + i)->m_CodewordsIdx + (pModel + i)->m_iNumEntries;
            TCodeword[top_index] = TCodewordCacheBook[(pCache + i)->m_CodewordsIdx + leavingIdx];
            (pModel + i)->m_iTotal = (pModel + i)->m_iTotal + 1;
            TCodeword[top_index].m_iT_first_time = (pModel + i)->m_iTotal;
            TCodeword[top_index].m_iT_last_time = (pModel + i)->m_iTotal;
            TCodeword[top_index].m_iMNRL = (pModel + i)->m_iTotal - 1;
            (pModel + i)->m_iNumEntries = (pModel + i)->m_iNumEntries + 1;
            k = 0;
            for (j = 0; j < (pCache + i)->m_iNumEntries; j++) {
                if (j == leavingIdx) continue;
                else {
                    int temp_idx = (pCache + i)->m_CodewordsIdx;
                    TTempCache[k] = TCodewordCacheBook[temp_idx];
                    k++;
                }
            }
            (pCache + i)->m_iNumEntries = k;
            for(j = 0; j < k; j++) {
                TCodeword[i] = TTempCache[i];
            }
        }
    }
}

void MultiCues::gpuTextureAbsorption(int absorpCnt, short *TContinuousCnt,
                                     short *TReferredIndex, TextureModel *TModel,
                                     TextureCodeword *TCodeword, TextureModel *TCacheBook,
                                     TextureCodeword *TCodewordCacheBook, bool * mask)
{
    dim3 blockDim(threadsPerBlock, threadsPerBlock);
    dim3 gridDim(ceil((float)m_ResizeWidth/threadsPerBlock), ceil((float)m_ResizeHeight/threadsPerBlock));

    cudaTextureAbsorption<<<gridDim, blockDim>>>(absorpCnt, TContinuousCnt, TReferredIndex,
                                                 TModel, TCodeword, TCacheBook, TCodewordCacheBook,
                                                 m_TCodewordTempCache, mask, m_neighborNum,
                                                 m_neighborRadius, m_ResizeWidth, m_ResizeHeight);
    gpuErrChk(cudaDeviceSynchronize());
}

__global__ void cudaColorAbsorption(int absorpCnt, short *CContinuousCnt,
                                    short *CReferredIndex, ColorModel *CModel,
                                    ColorCodeword *CCodeword, ColorModel *CCacheBook,
                                    ColorCodeword *CCodewordCacheBook, ColorCodeword * CTempCache,
                                    bool * mask, int radius, int rWidth, int rHeight)
{
    int row = threadIdx.y + IMUL(blockIdx.y, blockDim.y);
    int col = threadIdx.x + IMUL(blockIdx.x, blockDim.x);

    if(row < rHeight && col < rWidth)
    {
        int idx = col + IMUL(row, rWidth);
        if(mask != nullptr)
        {
            if(mask[idx] == false)
                return;
        }
        if(row < radius || col < radius || row >= rHeight - radius || col >= rWidth - radius)
            return;

        ColorModel * pModel = CModel + idx;
        ColorModel * pCache = CCacheBook + idx;
        int leavingIdx = CReferredIndex[idx];

        //array expansion
        if (pModel->m_iElementArraySize == pModel->m_iNumEntries) {
            pModel->m_iElementArraySize = pModel->m_iElementArraySize + 5;
            assert(pModel->m_iElementArraySize <= CUDA_MAX_CODEWORDS_SIZE);
        }

        //movement from the cache-book to the codebook
        int top_index = pModel->m_CodewordsIdx + pModel->m_iNumEntries;
        CCodeword[top_index] = CCodewordCacheBook[pCache->m_CodewordsIdx + leavingIdx];
        pModel->m_iTotal = pModel->m_iTotal + 1;
        CCodeword[top_index].m_iT_first_time = pModel->m_iTotal;
        CCodeword[top_index].m_iT_last_time = pModel->m_iTotal;
        CCodeword[top_index].m_iMNRL = pModel->m_iTotal - 1;

        pModel->m_iNumEntries = pModel->m_iNumEntries + 1;

        int k = 0;
        for (int i = 0; i < pCache->m_iNumEntries; i++) {
            if (i == leavingIdx) continue;
            else {
                CTempCache[k] = CCodewordCacheBook[pCache->m_CodewordsIdx + i];
                k++;
            }
        }
        for(int i = 0; i < k; i++)
            CCodeword[pCache->m_CodewordsIdx + i] = CTempCache[i];
        pCache->m_iNumEntries = k;

    }
}

void MultiCues::gpuColorAbsorption(int absorpCnt, short *CContinuousCnt,
                                   short *CReferredIndex, ColorModel *CModel,
                                   ColorCodeword *CCodeword, ColorModel *CCacheBook,
                                   ColorCodeword *CCodewordCacheBook, bool * mask)
{
    dim3 blockDim(threadsPerBlock, threadsPerBlock);
    dim3 gridDim(ceil((float)m_ResizeWidth/threadsPerBlock), ceil((float)m_ResizeHeight/threadsPerBlock));

    cudaColorAbsorption<<<gridDim, blockDim>>>(absorpCnt, CContinuousCnt, CReferredIndex,
                                               CModel, CCodeword, CCacheBook, CCodewordCacheBook,
                                               m_CCodewordTempCache, mask,
                                               m_neighborRadius, m_ResizeWidth, m_ResizeHeight);
    gpuErrChk(cudaDeviceSynchronize());

}

__global__ void cudaTextureClearNonEssentialEntriesForCacheBook(short nClearNum, unsigned char *landmarkMap,
                                                                short *TReferredIdx, TextureModel *TCacheBook,
                                                                TextureCodeword *TCodeword, int * keepCnt,
                                                                TextureCodeword * TCodewordTemp,
                                                                int neighborNum,
                                                                int rWidth, int rHeight)
{
    int row = threadIdx.y + IMUL(blockIdx.y, blockDim.y);
    int col = threadIdx.x + IMUL(blockIdx.x, blockDim.x);

    if(row < rHeight && col < rWidth)
    {
        int idx = col + IMUL(row, rWidth);
        TextureModel * pCacheBook = TCacheBook + IMUL(idx, neighborNum);

        TextureModel * aModel;
        short nReferredIdx;
        for (int n = 0; n < neighborNum; n++) {
            aModel = pCacheBook + n;
            nReferredIdx = (TReferredIdx + IMUL(idx, neighborNum))[n];

            //pCachebook->m_iTotal < nClearNum? --> MNRL update
            if (aModel->m_iTotal < nClearNum)
            {
                for (int i = 0; i < aModel->m_iNumEntries; i++)
                {
                    if (landmarkMap[idx] == 255 && i == nReferredIdx)
                        TCodeword[aModel->m_CodewordsIdx + i].m_iMNRL = 0;
                    else
                        TCodeword[aModel->m_CodewordsIdx + i].m_iMNRL++;
                }

                aModel->m_iTotal++;
            }

            //Perform clearing
            else {
                int iStaleThreshold = 5;
                int * aKeep = keepCnt + idx * CUDA_MAX_CODEWORDS_SIZE;
                short nKeepCnt;
                nKeepCnt = 0;

                for (int i = 0; i < aModel->m_iNumEntries; i++) {
                    if (TCodeword[aModel->m_CodewordsIdx + i].m_iMNRL < iStaleThreshold) {
                        aKeep[i] = 1;
                        nKeepCnt++;
                    }
                    else aKeep[i] = 0;
                }

                aModel->m_iElementArraySize = nKeepCnt + 2;
                if (aModel->m_iElementArraySize < 3)
                    aModel->m_iElementArraySize = 3;

                nKeepCnt = 0;
                TextureCodeword * temp = TCodewordTemp + idx * CUDA_MAX_CODEWORDS_SIZE;
                for (int i = 0; i < aModel->m_iNumEntries; i++) {
                    if (aKeep[i] == 1) {
                        temp[nKeepCnt] = TCodeword[aModel->m_CodewordsIdx + i];
                        temp[nKeepCnt].m_iMNRL = 0;
                        nKeepCnt++;
                    }
                }
                for(int i = 0; i < nKeepCnt; i++) {
                    TCodeword[aModel->m_CodewordsIdx + i] = temp[i];
                }

                aModel->m_iNumEntries = nKeepCnt;
                aModel->m_iTotal = 0;

            }
        }

    }
}

void MultiCues::gpuTextureClearNonEssentialEntriesForCacheBook(short nClearNum, unsigned char *landmarkMap,
                                                               short *TReferredIdx, TextureModel *TCacheBook,
                                                               TextureCodeword *TCodeword)
{
    dim3 blockDim(threadsPerBlock, threadsPerBlock);
    dim3 gridDim(ceil((float)m_ResizeWidth/threadsPerBlock), ceil((float)m_ResizeHeight/threadsPerBlock));

    cudaTextureClearNonEssentialEntriesForCacheBook<<<gridDim, blockDim>>>(nClearNum, landmarkMap, TReferredIdx, TCacheBook,
                                                                           TCodeword, m_TkeepCnt, m_TCodewordTemp, m_neighborNum,
                                                                           m_ResizeWidth, m_ResizeHeight);
    gpuErrChk(cudaDeviceSynchronize());
}

__global__ void cudaColorClearNonEssentialEntriesForCacheBook(short nClearNum, unsigned char *landmarkMap,
                                                              short *CRefferedIdx, ColorModel *CCacheBook,
                                                              ColorCodeword *CCodeword, int * keepCnt,
                                                              ColorCodeword *CCodewordTemp, int radius,
                                                              int rWidth, int rHeight)
{
    int row = threadIdx.y + IMUL(blockIdx.y, blockDim.y);
    int col = threadIdx.x + IMUL(blockIdx.x, blockDim.x);

    if(row < rHeight && col < rWidth)
    {
        if(row < radius || col < radius || row >= rHeight - radius || col >= rWidth - radius)
            return;
        int idx = col + IMUL(row, rWidth);
        ColorModel * pCacheBook = CCacheBook + idx;

        if (pCacheBook->m_iTotal < nClearNum) {
            for (int i = 0; i < pCacheBook->m_iNumEntries; i++) {
                if (landmarkMap[idx] == 255 && i == CRefferedIdx[idx])
                    CCodeword[pCacheBook->m_CodewordsIdx + i].m_iMNRL = 0;
                else
                    CCodeword[pCacheBook->m_CodewordsIdx + i].m_iMNRL++;
            }

            pCacheBook->m_iTotal++;
        }

        else
        {
            int iStaleThreshold = 5;
            int* aKeep = keepCnt + idx * CUDA_MAX_CODEWORDS_SIZE;;
            short nKeepCnt;
            nKeepCnt = 0;

            for (int i = 0; i < pCacheBook->m_iNumEntries; i++) {
                if (CCodeword[pCacheBook->m_CodewordsIdx + i].m_iMNRL < iStaleThreshold) {
                    aKeep[i] = 1;
                    nKeepCnt++;
                }
                else aKeep[i] = 0;
            }

            pCacheBook->m_iElementArraySize = nKeepCnt + 2;
            //		if (pCachebook->m_iElementArraySize < 3) pCachebook->m_iElementArraySize = 3;
            nKeepCnt = 0;
            ColorCodeword * temp = CCodewordTemp + idx * CUDA_MAX_CODEWORDS_SIZE;
            for (int i = 0; i < pCacheBook->m_iNumEntries; i++) {
                if (aKeep[i] == 1) {
                    temp[nKeepCnt] = CCodeword[pCacheBook->m_CodewordsIdx + i];
                    temp[nKeepCnt].m_iMNRL = 0;
                    nKeepCnt++;
                }
            }
            for(int i = 0; i < nKeepCnt; i++)
                CCodeword[pCacheBook->m_CodewordsIdx + i] = temp[i];

            //ending..
            pCacheBook->m_iNumEntries = nKeepCnt;
            pCacheBook->m_iTotal = 0;
        }
    }
}

void MultiCues::gpuColorClearNonEssentialEntriesForCacheBook(short nClearNum, unsigned char *landmarkMap,
                                                             short *CRefferedIdx, ColorModel *CCacheBook,
                                                             ColorCodeword *CCodeword)
{
    dim3 blockDim(threadsPerBlock, threadsPerBlock);
    dim3 gridDim(ceil((float)m_ResizeWidth/threadsPerBlock), ceil((float)m_ResizeHeight/threadsPerBlock));

    cudaColorClearNonEssentialEntriesForCacheBook<<<gridDim, blockDim>>>(nClearNum, landmarkMap, CRefferedIdx,
                                                  CCacheBook, CCodeword, m_CkeepCnt,
                                                  m_CCodewordTemp, m_neighborRadius,
                                                  m_ResizeWidth, m_ResizeHeight);
    gpuErrChk(cudaDeviceSynchronize());
}

__global__ void cudaCreateGaussianModel(GaussModel * gModel, unsigned char * gray,
                                        bool * mask, float learningRate,
                                        int gWidth, int gHeight,
                                        int rWidth, int rHeight, int blockSize)
{
    int row = threadIdx.y + IMUL(blockIdx.y, blockDim.y);
    int col = threadIdx.x + IMUL(blockIdx.x, blockDim.x);
    if(row < gHeight && col < gWidth)
    {
        int index = col + IMUL(row, gWidth);
        float mean = 0;
        int npixel = 0;
        float obs_var = gModel[index].var;
        for(int y = 0; y < blockSize; y++)
        {
            for(int x = 0; x < blockSize; x++)
            {
                int idx = y + row;
                int idy = x + col;
                int srcIdx = idx + IMUL(idy, rWidth);
                if(mask != nullptr)
                {
                    if(mask[srcIdx] == false)
                        continue;
                }
                if(idx < 0 || idx >= rWidth || idy < 0 || idy >= rHeight)
                {
                    continue;
                }
                mean += (float)gray[srcIdx];
                npixel++;

                float fDiff = (float)gray[srcIdx] - gModel[index].mean;
                float pixelDist = fDiff * fDiff;
                obs_var = (obs_var > pixelDist)? obs_var : pixelDist;
            }
        }
        if(npixel != 0)
        {
            mean = mean / npixel;
        }
        else
        {
            mean = 0;
        }
        gModel[index].mean = (1 - learningRate) * gModel[index].mean + learningRate * mean;
        gModel[index].var = (1 - learningRate) * gModel[index].var + learningRate * obs_var;
    }
}

void MultiCues::gpuCreateGaussianModel(float learningRate, bool * mask)
{
    dim3 blockDim(threadsPerBlock, threadsPerBlock);
    dim3 gridDim(ceil((float)m_gaussModelW / threadsPerBlock), ceil((float)m_gaussModelH/threadsPerBlock));

    cudaCreateGaussianModel<<<gridDim, blockDim>>>(m_dGaussianBgModel, m_dFilteredImg, mask, learningRate, m_gaussModelW,
                            m_gaussModelH, m_ResizeWidth, m_ResizeHeight, m_gaussBlockSize);
    gpuErrChk(cudaDeviceSynchronize());
}
}
