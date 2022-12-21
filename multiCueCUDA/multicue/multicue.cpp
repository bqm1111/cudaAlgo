#include "multicue.hpp"

namespace multiCue {
MultiCues::MultiCues()
{
    //----------------------------------
    //	User adjustable parameters
    //----------------------------------
    m_trainingPeriod = 5;											//the training period								(The parameter t in the paper)
    g_iT_ModelThreshold = 1;										//the threshold for texture-model based BGS.		(The parameter tau_T in the paper)
    g_iC_ModelThreshold = 10;										//the threshold for appearance based verification.  (The parameter tau_A in the paper)

    m_learningRate = 0.01f;											//the learning rate for background models.			(The parameter alpha in the paper)

    m_nTextureTrainVolRange = 5;									//the codebook size factor for texture models.		(The parameter k in the paper)
    m_nColorTrainVolRange = 20;										//the codebook size factor for color models.		(The parameter eta_1 in the paper)

    m_AbsorptionEnable = false;										//If true, cache-book is also modeled for ghost region removal.
    m_AbsorptionPeriod = 200;										//the period to absorb static ghost regions

    m_ResizeWidth = 160, m_ResizeHeight = 120;								//Frames are precessed after reduced in this size .
    //    m_ResizeWidth = 32, m_ResizeHeight = 24;								//Frames are precessed after reduced in this size

    //------------------------------------
    //	For codebook maintenance
    //------------------------------------
    m_backClearPeriod = 200;		//300								//the period to clear background models
    g_iCacheClearPeriod = 30;		//30								//the period to clear cache-book models

    //------------------------------------
    //	Initialization of other parameters
    //------------------------------------
    m_neighborNum = 6, m_neighborRadius = 2;
    //    m_confidenceThresh = g_iT_ModelThreshold / (float)m_neighborNum;	//the final decision threshold
    m_confidenceThresh = 0.15;

    m_frameCount = 0;
    m_fgMapEnable = false;									//true only when BGS is successful
    m_ModelMemAllocated = false;									//To handle memory..
    m_NonModelMemAllocated = false;								//To handle memory..

    m_gaussBlockSize = 2;
    h_init = false;
    THRES_BG_REFINE = 30;

    //    h_learningrate = 0.05;

    //  initLoadSaveConfig(algorithmName);
}

MultiCues::~MultiCues()
{
    releaseMem();
}

void MultiCues::allocateTextureModelRelatedMemory()
{
    int gridSize = m_ResizeHeight * m_ResizeWidth * m_neighborNum;
    // neigborhood system related
    gpuErrChk(cudaMalloc((void**)&m_neighborDirection, sizeof(short2) * gridSize));
    setNeighborDirection();

    int iElementArraySize = 6;

    gpuErrChk(cudaMalloc((void**)&m_TCodeword, sizeof(TextureCodeword) * CUDA_MAX_CODEWORDS_SIZE * gridSize));
    gpuErrChk(cudaMalloc((void**)&m_TextureModel, sizeof(TextureModel) * gridSize));
    allocateTextureModelRelatedMemoryHelper(m_TextureModel, iElementArraySize, 1, m_ResizeWidth, m_ResizeHeight, m_neighborNum);
    gpuErrChk(cudaMalloc((void**)&m_textureConfMap, sizeof(float) * m_ResizeHeight * m_ResizeWidth));
    gpuErrChk(cudaMalloc((void**)&m_TkeepCnt, sizeof(int) * gridSize *  CUDA_MAX_CODEWORDS_SIZE));

    gpuErrChk(cudaMalloc((void**)&m_TCodewordTemp, sizeof(TextureCodeword) * gridSize * CUDA_MAX_CODEWORDS_SIZE));
    gpuErrChk(cudaMalloc((void**)&m_TCodewordTempCache, sizeof(TextureCodeword) * gridSize * CUDA_MAX_CODEWORDS_SIZE));

    // cache-book related
    if(m_AbsorptionEnable == true)
    {
        iElementArraySize /= 2;
        if(iElementArraySize < 3)
            iElementArraySize = 3;
        gpuErrChk(cudaMalloc((void**)&m_codeWordTCacheBook, sizeof(TextureCodeword) * CUDA_MAX_CODEWORDS_SIZE * gridSize));
        gpuErrChk(cudaMalloc((void**)&m_TCacheBook, sizeof(TextureModel) * gridSize));
        allocateTextureModelRelatedMemoryHelper(m_TCacheBook, iElementArraySize, 0, m_ResizeWidth, m_ResizeHeight, m_neighborNum);


        gpuErrChk(cudaMalloc((void**)&m_TReferredIndex, sizeof(short) * gridSize));
        gpuErrChk(cudaMemset(m_TReferredIndex, -1, sizeof(short) * gridSize));
        gpuErrChk(cudaMalloc((void**)&m_TContinuousCnt, sizeof(short) * gridSize));
        gpuErrChk(cudaMemset(m_TContinuousCnt, 0, sizeof(short) * gridSize));
    }
}

void MultiCues::releaseTextureModelRelatedMemory()
{
    gpuErrChk(cudaFree(m_TCodeword));
    gpuErrChk(cudaFree(m_TextureModel));
    gpuErrChk(cudaFree(m_neighborDirection));
    gpuErrChk(cudaFree(m_textureConfMap));
    gpuErrChk(cudaFree(m_TkeepCnt));
    gpuErrChk(cudaFree(m_TCodewordTemp));
    gpuErrChk(cudaFree(m_TCodewordTempCache));

    if(m_AbsorptionEnable == true)
    {
        gpuErrChk(cudaFree(m_codeWordTCacheBook));
        gpuErrChk(cudaFree(m_TCacheBook));
        gpuErrChk(cudaFree(m_TReferredIndex));
        gpuErrChk(cudaFree(m_TContinuousCnt));
    }
}

void MultiCues::allocateColorModelRelatedMemory()
{
    int iElementArraySize = 10;
    int rImgSize = m_ResizeHeight * m_ResizeWidth;

    gpuErrChk(cudaMalloc((void**)&m_CCodeword, sizeof(ColorCodeword) * CUDA_MAX_CODEWORDS_SIZE * rImgSize));
    gpuErrChk(cudaMalloc((void**)&m_ColorModel, sizeof(ColorModel) * rImgSize));
    allocateColorModelRelatedMemoryHelper(m_ColorModel, iElementArraySize, 1, m_ResizeWidth, m_ResizeHeight);
    gpuErrChk(cudaMalloc((void**)&m_CkeepCnt, sizeof(int) * rImgSize * CUDA_MAX_CODEWORDS_SIZE));

    gpuErrChk(cudaMalloc((void**)&m_CCodewordTemp, sizeof(ColorCodeword) * rImgSize * CUDA_MAX_CODEWORDS_SIZE));
    gpuErrChk(cudaMalloc((void**)&m_CCodewordTempCache, sizeof(ColorCodeword) * rImgSize * CUDA_MAX_CODEWORDS_SIZE));

    if(m_AbsorptionEnable == true)
    {
        iElementArraySize = 1;
        gpuErrChk(cudaMalloc((void**)&m_codeWordCCacheBook, sizeof(ColorCodeword) * CUDA_MAX_CODEWORDS_SIZE * rImgSize));
        gpuErrChk(cudaMalloc((void**)&m_CCacheBook, sizeof(ColorModel) * rImgSize));
        allocateColorModelRelatedMemoryHelper(m_CCacheBook, iElementArraySize, 0, m_ResizeWidth, m_ResizeHeight);

        gpuErrChk(cudaMalloc((void**)&m_CReferredIndex, sizeof(short) * rImgSize));
        gpuErrChk(cudaMemset(m_CReferredIndex, -1, sizeof(short) * rImgSize));
        gpuErrChk(cudaMalloc((void**)&m_CContinuousCnt, sizeof(short) * rImgSize));
        gpuErrChk(cudaMemset(m_CContinuousCnt, 0, sizeof(short) * rImgSize));
    }
}

void MultiCues::releaseColorModelRelatedMemory()
{
    gpuErrChk(cudaFree(m_ColorModel));
    gpuErrChk(cudaFree(m_CCodeword));
    gpuErrChk(cudaFree(m_CkeepCnt));
    gpuErrChk(cudaFree(m_CCodewordTemp));
    gpuErrChk(cudaFree(m_CCodewordTempCache));

    if(m_AbsorptionEnable)
    {
        gpuErrChk(cudaFree(m_codeWordCCacheBook));
        gpuErrChk(cudaFree(m_CCacheBook));
        gpuErrChk(cudaFree(m_CReferredIndex));
        gpuErrChk(cudaFree(m_CContinuousCnt));
    }
}

void MultiCues::allocateGaussModelRelatedMemory()
{
    m_gaussModelW = m_ResizeWidth / m_gaussBlockSize;
    m_gaussModelH = m_ResizeHeight / m_gaussBlockSize;
    int gaussianGridSize = m_gaussModelH * m_gaussModelW;
    gpuErrChk(cudaHostAlloc((void**)&m_hGaussianBgModel, sizeof(GaussModel) * gaussianGridSize, cudaHostAllocMapped));
    gpuErrChk(cudaHostGetDevicePointer(reinterpret_cast<void**>(&m_dGaussianBgModel), m_hGaussianBgModel, 0));

    gpuErrChk(cudaMemset(m_dGaussianBgModel, 0, sizeof(GaussModel) * gaussianGridSize));

}

void MultiCues::releaseGaussModelRelatedMemory()
{
    gpuErrChk(cudaFreeHost(m_hGaussianBgModel));
}

void MultiCues::initParam(unsigned char *srcImg, int imgWidth, int imgHeight)
{
    // Initialize all parameters and allocate memories in the first frame
    m_width = imgWidth;
    m_height = imgHeight;

    // Release memory in the previous use
    releaseMem();
    int rImgSize = m_ResizeHeight * m_ResizeWidth;
    // Memory initialization
    gpuErrChk(cudaMalloc((void**)&m_dResizedImg, rImgSize * sizeof(uchar)));
    gpuErrChk(cudaHostAlloc((void**)&m_hFilteredImg, rImgSize * sizeof(uchar), cudaHostAllocMapped));
    gpuErrChk(cudaHostGetDevicePointer(reinterpret_cast<void**>(&m_dFilteredImg), m_hFilteredImg, 0));

    gpuErrChk(cudaMalloc((void**)&m_dlandmarkMap, rImgSize * sizeof(uchar)));
    gpuErrChk(cudaHostAlloc((void**)&m_hResizedFgMap, rImgSize * sizeof(uchar), cudaHostAllocMapped));
    gpuErrChk(cudaHostGetDevicePointer(reinterpret_cast<void**>(&m_dResizedFgMap), m_hResizedFgMap, 0));

    //    gpuErrChk(cudaMalloc((void**)&m_dResizedFgMap, rImgSize * sizeof(uchar)));
    gpuErrChk(cudaHostAlloc((void**)&m_hUpdateMap, rImgSize * sizeof(bool), cudaHostAllocMapped));
    gpuErrChk(cudaHostGetDevicePointer(reinterpret_cast<void**>(&m_dUpdateMap), m_hUpdateMap, 0));
    gpuErrChk(cudaMalloc((void**)&m_dUpdateMapCache, rImgSize * sizeof(bool)));

    // Bounding Box related
    int maxBoxNum = 300;
    gpuErrChk(cudaMallocHost((void**)&m_BboxInfo, sizeof(BoundingBoxInfo)));
    m_BboxInfo->boxNum = maxBoxNum;
    gpuErrChk(cudaHostAlloc((void**)&m_BboxInfo->hbox, maxBoxNum * sizeof(short4), cudaHostAllocMapped));
    gpuErrChk(cudaHostAlloc((void**)&m_BboxInfo->hRbox, maxBoxNum * sizeof(short4), cudaHostAllocMapped));
    gpuErrChk(cudaHostAlloc((void**)&m_BboxInfo->h_isValidBox, maxBoxNum * sizeof(bool), cudaHostAllocMapped));

    gpuErrChk(cudaHostGetDevicePointer(reinterpret_cast<void**>(&m_BboxInfo->dbox), m_BboxInfo->hbox, 0));
    gpuErrChk(cudaHostGetDevicePointer(reinterpret_cast<void**>(&m_BboxInfo->dRbox), m_BboxInfo->hRbox, 0));
    gpuErrChk(cudaHostGetDevicePointer(reinterpret_cast<void**>(&m_BboxInfo->d_isValidBox), m_BboxInfo->h_isValidBox, 0));

    //--------------------------------------------------------
    // texture model related
    //--------------------------------------------------------

    allocateTextureModelRelatedMemory();
    //--------------------------------------------------------
    // color moddel related
    //--------------------------------------------------------
    allocateColorModelRelatedMemory();
    //--------------------------------------------------------
    // Gauss moddel related
    //--------------------------------------------------------
#ifdef GAUSS
    allocateGaussModelRelatedMemory();
#endif
    m_ModelMemAllocated = true;
    m_NonModelMemAllocated = true;
}

void MultiCues::movingDetectObject(unsigned char *d_srcImg, unsigned char * h_srcImg, int srcWidth, int srcHeight)
{
    if(m_frameCount == 0)
    {
        getExeTime("Init Time = ", initParam(d_srcImg, srcWidth, srcHeight));
    }

    m_objRect.clear();
    process(d_srcImg, h_srcImg);

    //!!
    //! TODO: Get blob and verify candidate blob to get the final m_objRect
    //!

    if(m_fgMapEnable)
    {
        auto start = getMoment;
        cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
        cv::morphologyEx(m_fgMap, m_fgMap, cv::MORPH_OPEN, element);
        //        cv::imshow("ForegroundMap", m_fgMap);
        cv::Mat results, stats, centroids;
        int nConnectedComponents = cv::connectedComponentsWithStats(m_fgMap, results, stats, centroids, 4, CV_32S);
        for(int ilabel = 1; ilabel < nConnectedComponents; ilabel++)
        {
            int area = (stats.at<int>(ilabel, cv::CC_STAT_AREA)) ;
            cv::Rect rect;
            rect.x = stats.at<int>(ilabel, cv::CC_STAT_LEFT);
            rect.y = stats.at<int>(ilabel, cv::CC_STAT_TOP);
            rect.width  = stats.at<int>(ilabel, cv::CC_STAT_WIDTH);
            rect.height = stats.at<int>(ilabel, cv::CC_STAT_HEIGHT);
            float ratio = (float)rect.height / (float)rect.width;
            //                std::cout << "Rect = " << rect << " - area = " << area << std::endl;
            if(area > MIN_AREA && area < MAX_AREA && ratio > 0.1 && ratio < 4.0)
            {
                m_objRect.push_back(rect);
            }
        }
        auto end = getMoment;
        getTimeElapsed("Final Step Time = ", end, start);
    }
}

void MultiCues::process(unsigned char *d_img, unsigned char *h_img)
{

    //!!
    //! 1. Background Modelling
    //!
    printf("=================== frame Count = %d ================================\n", m_frameCount);

    if(m_frameCount <= m_trainingPeriod)
    {
        getExeTime("1. Background Modelling time = ", bgModelling(d_img));
    }
    //!!
    //! 2. Background Subtraction
    //!
    else
    {
        m_fgMapEnable = false;  // This will be set to true if BGS is successful
        getExeTime("1. fgExtraction Time = ", fgExtraction(d_img, h_img));

        getExeTime("1. updateModel Time = ", updateModel());
        // Get BGS result
        getExeTime("1. getFgMap Time = ", getFgMap());
    }
    m_frameCount++;
    firstTime = false;
}

void MultiCues::bgModelling(unsigned char *d_img)
{
    //!! Preprocessing Frame including
    //! 1. Resize input image
    //! 2. GaussianBlur with sigma = 0.7
    //! 3. RGB2HSV (can be ignored if input is gray image already(should be the case))
    getExeTime("-- 2. PreProcessing Time = ", preProcessing(d_img));

    float learningRate = m_learningRate * 4;

    //!!
    //! BackgroundModelling
    //!

    getExeTime("-- 2. Texture ModelConstruction Time = ",
               gpuTextureModelConstruction(m_nTextureTrainVolRange, learningRate,
                                           m_dFilteredImg, m_neighborDirection,
                                           m_TextureModel,
                                           m_TCodeword,
                                           m_TReferredIndex,
                                           m_TContinuousCnt,
                                           m_neighborRadius,
                                           m_ResizeWidth, m_ResizeHeight, m_neighborNum));

#ifdef printTexture
    printf("============ Texture Information ====================\n");
    int gridSize = m_ResizeHeight * m_ResizeWidth * m_neighborNum;
    TextureModel * h_TModel = (TextureModel *)malloc(sizeof(TextureModel) * gridSize);
    gpuErrChk(cudaMemcpy(h_TModel, m_TextureModel, gridSize * sizeof(TextureModel), cudaMemcpyDeviceToHost));

    printf("Texture Model iNumEntries= \n");
    for(int y = 0; y < m_ResizeHeight;y++)
    {
        for(int x = 0; x < m_ResizeWidth; x++)
        {
            for(int k = 0; k < m_neighborNum; k++)
            {
                int idx = (y * m_ResizeWidth + x) * m_neighborNum + k;
                printf("%d\t", h_TModel[idx].m_iNumEntries);
            }
        }
        printf("\n");
    }

    printf("Texture Codeword Mean = \n");
    TextureCodeword * h_TCodeword = (TextureCodeword *)malloc(sizeof(TextureCodeword) * gridSize * CUDA_MAX_CODEWORDS_SIZE);
    gpuErrChk(cudaMemcpy(h_TCodeword, m_TCodeword, gridSize * CUDA_MAX_CODEWORDS_SIZE * sizeof(TextureCodeword), cudaMemcpyDeviceToHost));

    for(int y = 0; y < m_ResizeHeight; y++)
    {
        for(int x = 0; x < m_ResizeWidth; x++)
        {
            for(int k = 0; k < m_neighborNum; k++)
            {
                int idx = (y * m_ResizeWidth + x) * m_neighborNum + k;
                for(int j = 0; j < h_TModel[idx].m_iNumEntries; j++)
                {
                    printf("%f\t", h_TCodeword[idx * CUDA_MAX_CODEWORDS_SIZE + j].m_fMean);
                }
            }
        }
        printf("\n");
    }
    //    printf("Texture Codeword Low Thresh = \n");

    //    for(int y = 0; y < m_ResizeHeight; y++)
    //    {
    //        for(int x = 0; x < m_ResizeWidth; x++)
    //        {
    //            for(int k = 0; k < m_neighborNum; k++)
    //            {
    //                int idx = (y * m_ResizeWidth + x) * m_neighborNum + k;
    //                for(int j = 0; j < h_TModel[idx].m_iNumEntries; j++)
    //                {
    //                    printf("%f\t", h_TCodeword[idx * CUDA_MAX_CODEWORDS_SIZE + j].m_fLowThre);
    //                }
    //            }
    //        }
    //        printf("\n");
    //    }

    //    printf("Texture Codeword High Thresh = \n");

    //    for(int y = 0; y < m_ResizeHeight; y++)
    //    {
    //        for(int x = 0; x < m_ResizeWidth; x++)
    //        {
    //            for(int k = 0; k < m_neighborNum; k++)
    //            {
    //                int idx = (y * m_ResizeWidth + x) * m_neighborNum + k;
    //                for(int j = 0; j < h_TModel[idx].m_iNumEntries; j++)
    //                {
    //                    printf("%f\t", h_TCodeword[idx * CUDA_MAX_CODEWORDS_SIZE + j].m_fHighThre);
    //                }
    //            }
    //        }
    //        printf("\n");
    //    }
#endif
    getExeTime("-- 2. Color CodeBookConstruction Time = ",
               gpuColorCodeBookConstruction(m_nColorTrainVolRange, learningRate,
                                            m_dFilteredImg,
                                            m_ColorModel,
                                            m_CCodeword,
                                            m_CReferredIndex,
                                            m_CContinuousCnt,
                                            m_neighborRadius,
                                            m_ResizeWidth, m_ResizeHeight));
#ifdef printColor

    printf("============ Color Information ======================\n");
    int rImgSize = m_ResizeHeight * m_ResizeWidth;
    ColorModel * h_CModel = (ColorModel *)malloc(sizeof(ColorModel) * rImgSize);
    gpuErrChk(cudaMemcpy(h_CModel, m_ColorModel, rImgSize * sizeof(ColorModel), cudaMemcpyDeviceToHost));

    printf("Color Model iNumEntries = \n");
    for(int y = 0; y < m_ResizeHeight;y++)
    {
        for(int x = 0; x < m_ResizeWidth; x++)
        {
            int idx = (y * m_ResizeWidth + x) ;
            printf("%d\t", h_CModel[idx].m_iNumEntries);

        }
        printf("\n");
    }

    ColorCodeword * h_CCodeword = (ColorCodeword*)malloc(sizeof(ColorCodeword) * rImgSize * CUDA_MAX_CODEWORDS_SIZE);
    gpuErrChk(cudaMemcpy(h_CCodeword, m_CCodeword, rImgSize * CUDA_MAX_CODEWORDS_SIZE * sizeof(ColorCodeword), cudaMemcpyDeviceToHost));

    printf("Color Codeword Mean = \n");
    for(int y = 0; y< m_ResizeHeight;y++)
    {
        for(int x = 0; x < m_ResizeWidth; x++)
        {
            int idx = (y * m_ResizeWidth + x);
            for(int k = 0; k < h_CModel[idx].m_iNumEntries; k++)
            {
                printf("%f\t", h_CCodeword[idx*CUDA_MAX_CODEWORDS_SIZE + k].m_dMean);
            }
        }
        printf("\n");
    }
#endif
    if(m_frameCount == m_trainingPeriod)
    {
        getExeTime("-- 2. Texture Clear Time = ", gpuTextureClearNonEssentialEntries(m_trainingPeriod, m_TextureModel, m_TCodeword));
        getExeTime("-- 2. Color Clear Time = ", gpuColorClearNonEssentialEntries(m_trainingPeriod, m_ColorModel, m_CCodeword));
        //        m_frameCount++;
    }

#ifdef GAUSS
    gpuCreateGaussianModel(0.35);
#endif
}

void MultiCues::fgExtraction(unsigned char *d_img, unsigned char * h_img)
{
    getExeTime("-- 2. PreProcessing Time = ", preProcessing(d_img));

    getExeTime("-- 2. GetConfidenceMap Time = ", gpuGetConfidenceMap(m_dFilteredImg, m_textureConfMap, m_neighborDirection, m_TextureModel, m_TCodeword));

    getExeTime("-- 2. CreateLandMarkArray Time = ", gpuCreateLandMarkArray(m_confidenceThresh, m_nColorTrainVolRange, m_textureConfMap, m_dFilteredImg,
                                                                           m_neighborDirection, m_TextureModel, m_ColorModel, m_TCodeword,
                                                                           m_CCodeword, m_dlandmarkMap));


//    cv::Mat landmarkMap(m_ResizeHeight, m_ResizeWidth, CV_8UC1);
//    gpuErrChk(cudaMemcpy(landmarkMap.data, m_dlandmarkMap, m_ResizeHeight * m_ResizeWidth * sizeof(unsigned char), cudaMemcpyDeviceToHost));
//    cv::imshow("landmark", landmarkMap);

    getExeTime("-- 2. PostProcessing Time = ", gpuPostProcessing(h_img));
}

void MultiCues::updateModel()
{
    // Step 1: Update map construction
    gpuErrChk(cudaMemset(m_dUpdateMap, true, sizeof(bool) * m_ResizeHeight * m_ResizeWidth));
    gpuErrChk(cudaMemset(m_dUpdateMapCache, false, sizeof(bool) * m_ResizeHeight * m_ResizeWidth));

    getExeTime("gpuUpdateModel Time = ", gpuUpdateModel(m_BboxInfo, m_dResizedFgMap, m_dUpdateMap, m_dUpdateMapCache));
    // Step 2: Update Model

    gpuTextureModelConstruction(m_nTextureTrainVolRange, m_learningRate, m_dFilteredImg, m_neighborDirection,
                                m_TextureModel, m_TCodeword, m_TReferredIndex, m_TContinuousCnt, m_neighborRadius,
                                m_ResizeWidth, m_ResizeHeight, m_neighborNum, m_dUpdateMap);
    gpuColorCodeBookConstruction(m_nColorTrainVolRange, m_learningRate, m_dFilteredImg, m_ColorModel,
                                 m_CCodeword, m_CReferredIndex, m_CContinuousCnt, m_neighborRadius,
                                 m_ResizeWidth, m_ResizeHeight, m_dUpdateMap);

    // Clear non-essential codeword
    gpuTextureClearNonEssentialEntries(m_backClearPeriod, m_TextureModel, m_TCodeword, m_dUpdateMap);
    gpuColorClearNonEssentialEntries(m_backClearPeriod, m_ColorModel, m_CCodeword, m_dUpdateMap);

    if(m_AbsorptionEnable == true)
    {
        gpuTextureModelConstruction(m_nTextureTrainVolRange, m_learningRate, m_dFilteredImg,
                                    m_neighborDirection, m_TextureModel,m_TCodeword, m_TReferredIndex,
                                    m_TContinuousCnt, m_neighborRadius, m_ResizeWidth, m_ResizeHeight,
                                    m_neighborNum, m_dUpdateMapCache);
        gpuColorCodeBookConstruction(m_nColorTrainVolRange, m_learningRate, m_dFilteredImg, m_ColorModel,
                                     m_CCodeword, m_CReferredIndex, m_CContinuousCnt, m_neighborRadius,
                                     m_ResizeWidth, m_ResizeHeight, m_dUpdateMapCache);

        gpuTextureAbsorption(m_AbsorptionPeriod, m_TContinuousCnt,
                             m_TReferredIndex, m_TextureModel, m_TCodeword,
                             m_TCacheBook, m_codeWordTCacheBook, m_dUpdateMapCache);

        gpuColorAbsorption(m_AbsorptionPeriod, m_CContinuousCnt,
                           m_CReferredIndex, m_ColorModel,
                           m_CCodeword,m_CCacheBook, m_codeWordCCacheBook, m_dUpdateMapCache);
    }

    if(m_AbsorptionEnable == true)
    {
        gpuTextureClearNonEssentialEntriesForCacheBook(10, m_dlandmarkMap, m_TReferredIndex,
                                                       m_TCacheBook, m_TCodeword);
        gpuColorClearNonEssentialEntriesForCacheBook(10, m_dlandmarkMap, m_CReferredIndex,
                                                     m_CCacheBook, m_CCodeword);
    }
#ifdef GAUSS
    gpuCreateGaussianModel(0.35, m_dUpdateMap);
#endif
}

void MultiCues::getFgMap()
{
    if(m_fgMapEnable == false)
        return;
    cv::Mat temp(m_ResizeHeight, m_ResizeWidth, CV_8UC1, m_hResizedFgMap);
    cv::resize(temp, m_fgMap, cv::Size(m_width, m_height));
//    cv::imshow("fgMap", m_fgMap);
}

void MultiCues::preProcessing(unsigned char *d_img)
{
    getExeTime("Resize time = ", gpuResize(d_img, m_dResizedImg, m_width, m_height, m_ResizeWidth, m_ResizeHeight));
    getExeTime("GaussianBlur Time = ", gpuGaussianBlur(m_dResizedImg, m_dFilteredImg, m_ResizeWidth, m_ResizeHeight, 0.7));
}
void MultiCues::releaseMem()
{
    if(m_ModelMemAllocated == false && m_NonModelMemAllocated == false)
        return;

    if(m_ModelMemAllocated == true)
    {
        //!!
        //! TODO: Free bunch of thing here
        //!

        releaseTextureModelRelatedMemory();
        releaseColorModelRelatedMemory();
        releaseGaussModelRelatedMemory();
        m_ModelMemAllocated = false;
    }

    if(m_NonModelMemAllocated == true)
    {
        //!!
        //! TODO: Free bunch of thing here
        //!

        gpuErrChk(cudaFree(m_dResizedImg));
        gpuErrChk(cudaFree(m_dFilteredImg));
        gpuErrChk(cudaFree(m_dlandmarkMap));
        gpuErrChk(cudaFreeHost(m_hResizedFgMap));
        gpuErrChk(cudaFree(m_dUpdateMap));
        gpuErrChk(cudaFree(m_dUpdateMapCache));

        gpuErrChk(cudaFreeHost(m_BboxInfo->hbox));
        gpuErrChk(cudaFreeHost(m_BboxInfo->hRbox));
        gpuErrChk(cudaFreeHost(m_BboxInfo->h_isValidBox));
        gpuErrChk(cudaFreeHost(m_BboxInfo));

        m_NonModelMemAllocated = false;
    }
}

void MultiCues::gpuGaussianBlur(const unsigned char *d_src, unsigned char *d_dst, int width, int height, double sigma)
{
    sigma = 0.7;
    cv::cuda::GpuMat srcGpuMat(height, width, CV_8UC1, (void*)d_src);
    cv::cuda::GpuMat dstGpuMat(height, width, CV_8UC1, d_dst);

    cv::Ptr<cv::cuda::Filter>filter = cv::cuda::createGaussianFilter(srcGpuMat.type(),
                                                                     dstGpuMat.type(),
                                                                     cv::Size(7, 7),
                                                                     sigma);
    filter->apply(srcGpuMat, dstGpuMat);
    //    cv::Mat ret;
    //    cv::Mat resizeFrame;
    //    srcGpuMat.download(resizeFrame);
    //    dstGpuMat.download(ret);
    //    std::cout << "resizeFrame =\n" << resizeFrame << std::endl;
    //    std::cout << "blurredImg = \n" << ret << std::endl;
    return;
}

void MultiCues::getBoundingBox()
{
    cv::Mat stats, centroid, labelImg;
    cv::Mat resizeFgMap(m_ResizeHeight, m_ResizeWidth, CV_8UC1, m_hResizedFgMap);
    int nLabels;
    getExeTime("xxxxxxxx 5. Connected Component Time = ", nLabels= cv::connectedComponentsWithStats(resizeFgMap, labelImg, stats, centroid, 4, CV_32S));
    m_BboxInfo->boxNum = nLabels - 1;
    for(int i = 0; i < nLabels - 1; i++)
    {
        m_BboxInfo->hRbox[i].x = stats.at<int>(i, 0);
        m_BboxInfo->hRbox[i].y = stats.at<int>(i, 0) + stats.at<int>(i, 2);
        m_BboxInfo->hRbox[i].z = stats.at<int>(i, 1);
        m_BboxInfo->hRbox[i].w = stats.at<int>(i, 1) + stats.at<int>(i, 3);
        m_BboxInfo->h_isValidBox[i] = true;
    }
    float dH_ratio = (float)m_height / m_ResizeHeight;
    float dW_ratio = (float)m_width / m_ResizeWidth;

    for(int i = 0; i < m_BboxInfo->boxNum; i++)
    {
        m_BboxInfo->hbox[i].x = m_BboxInfo->hRbox[i].x * dW_ratio;
        m_BboxInfo->hbox[i].y = m_BboxInfo->hRbox[i].y * dH_ratio;
        m_BboxInfo->hbox[i].z = m_BboxInfo->hRbox[i].z * dW_ratio;
        m_BboxInfo->hbox[i].w = m_BboxInfo->hRbox[i].w * dH_ratio;
    }
}
void MultiCues::evaluateBoxSize()
{
    // Set threshold
    int iLowThreshold_w, iHighThreshold_w;
    iLowThreshold_w = m_ResizeWidth / 32; if (iLowThreshold_w < 5) iLowThreshold_w = 5;
    iHighThreshold_w = m_ResizeWidth - 5 ;

    int iLowThreshold_h, iHighThreshold_h;
    iLowThreshold_h = m_ResizeHeight / 24; if (iLowThreshold_h < 5) iLowThreshold_h = 5;
    iHighThreshold_h = m_ResizeHeight - 5;

    int iBoxWidth, iBoxHeight;
    // Perform verification
    for(int i = 0; i < m_BboxInfo->boxNum; i++)
    {
        iBoxWidth = m_BboxInfo->hRbox[i].y - m_BboxInfo->hRbox[i].x;
        iBoxHeight = m_BboxInfo->hRbox[i].w - m_BboxInfo->hRbox[i].z;
        if(iLowThreshold_w <= iBoxWidth && iBoxWidth <= iHighThreshold_w &&
                iLowThreshold_h <= iBoxHeight && iBoxHeight <= iHighThreshold_h)
        {
            m_BboxInfo->h_isValidBox[i] = true;
        }
        else
        {
            m_BboxInfo->h_isValidBox[i] = false;
        }
    }
}

void MultiCues::calcOverlap()
{
    int boxNum = m_BboxInfo->boxNum;
    for(int i = 0; i < boxNum; i++)
    {
        if(m_BboxInfo->h_isValidBox[i] == true)
        {
            for(int j=i; j < boxNum; j++)
            {
                if(i == j || m_BboxInfo->h_isValidBox[j] == false)
                {
                    continue;
                }
                int xmin = std::max(m_BboxInfo->hbox[i].x, m_BboxInfo->hbox[j].x);
                int xmax = std::min(m_BboxInfo->hbox[i].y, m_BboxInfo->hbox[j].y);
                int ymin = std::max(m_BboxInfo->hbox[i].z, m_BboxInfo->hbox[j].z);
                int ymax = std::min(m_BboxInfo->hbox[i].w, m_BboxInfo->hbox[j].w);
                int areaJ = (m_BboxInfo->hbox[j].x - m_BboxInfo->hbox[j].y) * (m_BboxInfo->hbox[j].z - m_BboxInfo->hbox[j].w);
                int inteArea = std::max(0, xmax - xmin ) * std::max(0 , ymax - ymin );
                int areaI = (m_BboxInfo->hbox[i].x - m_BboxInfo->hbox[i].y) * (m_BboxInfo->hbox[i].z - m_BboxInfo->hbox[i].w);

                float ratioAreaJ = (float)inteArea / float(areaJ);
                if(ratioAreaJ >= 0.6)
                {
                    m_BboxInfo->h_isValidBox[j] = false;

                }
                float ratioAreaI =  (float)inteArea / float(areaI);
                if(ratioAreaI >= 0.6)
                {
                    m_BboxInfo->h_isValidBox[i] = false;
                }
            }
        }
    }
}

void MultiCues::boundingBoxVerification(unsigned char * h_img)
{
    // Verify by the bouding box Size
    getExeTime("****** 4. EvaluateBoxSize Time = ", evaluateBoxSize());

    // verify by calculating overlap region
    getExeTime("****** 4. CalculateOverlap Time = ", calcOverlap());

    // Verify by checking whether the region is ghost or not
    getExeTime("****** 4. EvaluateGhostTime = ", evaluateGhostRegion(h_img));

    // Counting number of valid boxes
    m_fgNum = 0;
    for(int i = 0; i < m_BboxInfo->boxNum; i++)
    {
        if(m_BboxInfo->h_isValidBox[i] == true)
            m_fgNum++;
    }
}

void MultiCues::removingInvalidFgRegion()
{
    for(int k = 1; k < m_BboxInfo->boxNum; k++)
    {
        if(m_BboxInfo->h_isValidBox[k] == false)
        {
            for(int i = m_BboxInfo->hRbox[k].z; i < m_BboxInfo->hRbox[k].w; i++)
            {
                for(int j = m_BboxInfo->hRbox[k].x; j < m_BboxInfo->hRbox[k].y; j++)
                {
                    if(m_hResizedFgMap[i * m_ResizeWidth + j] == 255)
                        m_hResizedFgMap[i * m_ResizeWidth + j] = 0;
                }
            }
        }
    }
}

void MultiCues::gaussianRefineBgModel(unsigned char *h_img)
{
    if(m_BboxInfo->boxNum > 0)
    {
        for(int i = 0; i < m_BboxInfo->boxNum; i++)
        {
            if(m_BboxInfo->h_isValidBox[i] == true)
            {
                int npixel = 0;
                int xBox = m_BboxInfo->hRbox[i].x;
                int yBox = m_BboxInfo->hRbox[i].z;
                int wBox = m_BboxInfo->hRbox[i].y - m_BboxInfo->hRbox[i].x;
                int hBox = m_BboxInfo->hRbox[i].w - m_BboxInfo->hRbox[i].z;
                for(int y = yBox; y < yBox + hBox; y++)
                {
                    for(int x = xBox; x < xBox + wBox; x++)
                    {
                        int tlX = x / m_gaussBlockSize;
                        int tlY = y / m_gaussBlockSize;
                        if(tlX >= 0 && tlX < m_gaussModelW && tlY >= 0 && tlY < m_gaussModelH)
                        {
                            float fDiff = (float)h_img[y * m_ResizeWidth + x] - m_hGaussianBgModel[tlY * m_gaussModelW + tlX].mean;
                            float pixelDist = fDiff * fDiff;
                            if(pixelDist < THRES_BG_REFINE * m_hGaussianBgModel[tlY * m_gaussModelW + tlX].var)
                                npixel++;
                        }
                    }
                }

                if(float(npixel / (hBox * wBox)) > 0.1)
                {
                    m_BboxInfo->h_isValidBox[i] = false;
                    for(int y = yBox; y < yBox + hBox; y++)
                    {
                        for(int x = xBox; x < xBox + wBox; x++)
                        {
                            if(m_hResizedFgMap[y * m_ResizeWidth + x] == 255)
                                m_hUpdateMap[y * m_ResizeWidth + x] = true;
                        }
                    }
                }
            }
        }
    }
}

void MultiCues::evaluateGhostRegion(unsigned char * h_img)
{
    auto start = getMoment;
    cv::Mat srcImg(m_height, m_width, CV_8UC1, h_img);
    IplImage * frame = new IplImage(srcImg);

    double dThreshold = 15;

    for(int i = 0; i < m_BboxInfo->boxNum; i++)
    {
        if(m_BboxInfo->h_isValidBox[i] == true)
        {
            int iWin_w = m_BboxInfo->hRbox[i].y - m_BboxInfo->hRbox[i].x;
            int iWin_h = m_BboxInfo->hRbox[i].w - m_BboxInfo->hRbox[i].z;
            //Generating edge image from bound-boxed frame region
            IplImage* resized_frame = cvCreateImage(cvSize(m_ResizeWidth, m_ResizeHeight), IPL_DEPTH_8U, 1);
            cvResize(frame, resized_frame, CV_INTER_NN);

            cvSetImageROI(resized_frame, cvRect(m_BboxInfo->hRbox[i].x, m_BboxInfo->hRbox[i].z, iWin_w, iWin_h));
            IplImage* edge_frame = cvCreateImage(cvSize(iWin_w, iWin_h), IPL_DEPTH_8U, 1);

            cvCopy(resized_frame, edge_frame);
            cvCanny(edge_frame, edge_frame, 100, 150);

            //Generating edge image from aResForeMap
            IplImage* edge_fore = cvCreateImage(cvSize(iWin_w, iWin_h), IPL_DEPTH_8U, 1);
            for (int m = m_BboxInfo->hRbox[i].z; m < m_BboxInfo->hRbox[i].w; m++) {
                for (int n = m_BboxInfo->hRbox[i].x; n < m_BboxInfo->hRbox[i].y; n++) {
                    edge_fore->imageData[(m - m_BboxInfo->hRbox[i].z)*edge_fore->widthStep + (n - m_BboxInfo->hRbox[i].x)]
                            = (char)m_hResizedFgMap[m * m_ResizeWidth + n];
                }
            }

            cvCanny(edge_fore, edge_fore, 100, 150);

            //Calculating partial undirected Hausdorff distance
            double distance;
            distance = CalculateHausdorffDist(edge_frame, edge_fore);
            //Recording evaluation result
            if (distance > dThreshold)
            {

                for (int m = m_BboxInfo->hRbox[i].z; m < m_BboxInfo->hRbox[i].w; m++)
                {
                    for (int n = m_BboxInfo->hRbox[i].x; n < m_BboxInfo->hRbox[i].y; n++)
                    {
                        m_hUpdateMap[m * m_ResizeWidth + n] = true;
                    }
                }
                m_BboxInfo->h_isValidBox[i] = false;
            }
            cvResetImageROI(resized_frame);
            cvReleaseImage(&resized_frame);
            cvReleaseImage(&edge_frame);
            cvReleaseImage(&edge_fore);
        }
    }

    auto end = getMoment;
    getTimeElapsed("Step 1 Time = ", end, start);
#ifdef GAUSS
    gaussianRefineBgModel(m_hFilteredImg);
#endif
    // Step2: Adding information fo ghost region pixels to background model

    float fLearningRate = m_learningRate;

    gpuTextureModelConstruction(m_nTextureTrainVolRange, fLearningRate, m_dFilteredImg,
                                m_neighborDirection, m_TextureModel, m_TCodeword,
                                m_TReferredIndex, m_TContinuousCnt,
                                m_neighborRadius, m_ResizeWidth, m_ResizeHeight, m_neighborNum);

    gpuColorCodeBookConstruction(m_nColorTrainVolRange, fLearningRate, m_dFilteredImg,
                                 m_ColorModel, m_CCodeword,
                                 m_CReferredIndex, m_CContinuousCnt,
                                 m_neighborRadius, m_ResizeWidth,
                                 m_ResizeHeight);

    gpuTextureClearNonEssentialEntries(m_backClearPeriod, m_TextureModel, m_TCodeword);
    gpuColorClearNonEssentialEntries(m_backClearPeriod, m_ColorModel, m_CCodeword);
}
//-----------------------------------------------------------------------------------------------------------------------------------------//
//								the function to calculate partial undirected Hausdorff distance(forward distance)						   //																							   //
//-----------------------------------------------------------------------------------------------------------------------------------------//
double MultiCues::CalculateHausdorffDist(IplImage* input_image, IplImage* model_image) {

    //Step1: Generating imag vectors
    //For reduce errors, points at the image boundary are excluded
    std::vector<short2> vInput, vModel;
    short2 temp;

    //input image --> input vector
    for (int i = 0; i < input_image->height; i++) {
        for (int j = 0; j < input_image->width; j++) {

            if ((uchar)input_image->imageData[i*input_image->widthStep + j] == 0) continue;

            temp.x = j; temp.y = i;
            vInput.push_back(temp);
        }
    }
    //model image --> model vector
    for (int i = 0; i < model_image->height; i++) {
        for (int j = 0; j < model_image->width; j++) {
            if ((uchar)model_image->imageData[i*model_image->widthStep + j] == 0) continue;

            temp.x = j; temp.y = i;
            vModel.push_back(temp);
        }
    }

    if (vInput.empty() && !vModel.empty()) return (double)vModel.size();
    else if (!vInput.empty() && vModel.empty()) return (double)vInput.size();
    else if (vInput.empty() && vModel.empty()) return 0.0;

    //Step2: Calculating forward distance h(Model,Image)
    double dDist, temp1, temp2, dMinDist;
    std::vector<double> vTempDist;

    for (auto iter_m = vModel.begin(); iter_m < vModel.end(); iter_m++) {

        dMinDist = 9999999;
        for (auto iter_i = vInput.begin(); iter_i < vInput.end(); iter_i++) {
            temp1 = (*iter_m).x - (*iter_i).x;
            temp2 = (*iter_m).y - (*iter_i).y;
            dDist = temp1 * temp1 + temp2 * temp2;

            if (dDist < dMinDist) dMinDist = dDist;
        }
        vTempDist.push_back(dMinDist);
    }
    sort(vTempDist.begin(), vTempDist.end()); //in ascending order

    double dQuantileVal = 0.9, dForwardDistance;
    int iDistIndex = (int)(dQuantileVal*vTempDist.size());
    if (iDistIndex == (int)vTempDist.size()) iDistIndex -= 1;

    dForwardDistance = sqrt(vTempDist[iDistIndex]);
    return dForwardDistance;
}

void MultiCues::gpuPostProcessing(unsigned char * h_img)
{
    getExeTime("++++ 3. Morphological Time = ", gpuMorphologicalOperation(m_dlandmarkMap, m_dResizedFgMap, 0.5, 5, m_ResizeWidth, m_ResizeHeight));

    m_fgMapEnable = true;
    getExeTime("++++ 3. getBoundingBox Time = ", getBoundingBox());

    getExeTime("++++ 3. BoundingBoxVerification Time = ", boundingBoxVerification(h_img));

    getExeTime("++++ 3. RemovingInvalidFgRegion Time = ", removingInvalidFgRegion());

//        gpuLabelling(m_dResizedFgMap, d_labelCnt, d_labelTable);
}

}
