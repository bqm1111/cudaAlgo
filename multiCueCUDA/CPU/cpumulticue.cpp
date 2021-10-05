#include "cpumulticue.hpp"

using namespace multiCues;
using namespace cv;
using namespace std;

#define MIN3(x,y,z)  ((y) <= (z) ? ((x) <= (y) ? (x) : (y)) : ((x) <= (z) ? (x) : (z)))
#define MAX3(x,y,z)  ((y) >= (z) ? ((x) >= (y) ? (x) : (y)) : ((x) >= (z) ? (x) : (z)))

#ifndef PI
#define PI 3.141592653589793f
#endif
string DEBUGPATH = "/home/huongnt/Workspace/MovDet/src/multicues/build/debug";

MultiCues::MultiCues()
{
    //----------------------------------
    //	User adjustable parameters
    //----------------------------------
    g_iTrainingPeriod = 5;											//the training period								(The parameter t in the paper)
    g_iT_ModelThreshold = 1;										//the threshold for texture-model based BGS.		(The parameter tau_T in the paper)
    g_iC_ModelThreshold = 10;										//the threshold for appearance based verification.  (The parameter tau_A in the paper)

    g_fLearningRate = 0.01f;											//the learning rate for background models.			(The parameter alpha in the paper)

    g_nTextureTrainVolRange = 5;									//the codebook size factor for texture models.		(The parameter k in the paper)
    g_nColorTrainVolRange = 20;										//the codebook size factor for color models.		(The parameter eta_1 in the paper)

    g_bAbsorptionEnable = true;										//If true, cache-book is also modeled for ghost region removal.
    g_iAbsortionPeriod = 200;										//the period to absorb static ghost regions

//    g_iRWidth = 160, g_iRHeight = 120;								//Frames are precessed after reduced in this size .
        g_iRWidth = 32, g_iRHeight = 24;								//Frames are precessed after reduced in this size .

    //------------------------------------
    //	For codebook maintenance
    //------------------------------------
    g_iBackClearPeriod = 200;		//300								//the period to clear background models
    g_iCacheClearPeriod = 30;		//30								//the period to clear cache-book models

    //------------------------------------
    //	Initialization of other parameters
    //------------------------------------
    g_nNeighborNum = 6, g_nRadius = 2;
    g_fConfidenceThre = g_iT_ModelThreshold / (float)g_nNeighborNum;	//the final decision threshold

    g_iFrameCount = 0;
    g_bForegroundMapEnable = false;									//true only when BGS is successful
    g_bModelMemAllocated = false;									//To handle memory..
    g_bNonModelMemAllocated = false;								//To handle memory..

    h_blocksize = 2;
    h_init = false;
    THRES_BG_REFINE =30;

    //    h_learningrate = 0.05;

    //  initLoadSaveConfig(algorithmName);
}

MultiCues::~MultiCues() {
    Destroy();
}

//-----------------------------------------------------------------------------------------------------------------------------------------//
//											the main function to background modeling and subtraction									   //																   //
//-----------------------------------------------------------------------------------------------------------------------------------------//
void MultiCues::process(const cv::Mat &img_input, cv::Mat &img_output, cv::Mat &img_bgmodel, cv::Mat& H)
{
    init(img_input, img_output, img_bgmodel);

    //--STep1: Background Modeling--//
    IplImage* frame = new IplImage(img_input);
    IplImage* result_image = cvCreateImage(cvGetSize(frame), IPL_DEPTH_8U, 1);
    cvSetZero(result_image);
    if (g_iFrameCount <= g_iTrainingPeriod)
    {
        getExeTime("BackgroundModelling Time = ", BackgroundModeling_Par(frame));
        g_iFrameCount++;
    }
    //--Step2: Background Subtraction--//
    else {
        g_bForegroundMapEnable = false;
        getExeTime("ForegroundExtraction Time = ", ForegroundExtraction(frame));
        getExeTime("UpdateModel_Par Time = ", UpdateModel_Par());
        //Get BGS Results
        getExeTime("GetForeGroundMap Time = ", GetForegroundMap(result_image, NULL));
    }
    delete frame;

    img_background = cv::Mat::zeros(img_input.size(), img_input.type());
    img_foreground = cv::cvarrToMat(result_image, true);
    string img_fg_name = DEBUGPATH + "/fg.jpg";
    imwrite(img_fg_name, img_foreground);
    cvReleaseImage(&result_image);


    img_foreground.copyTo(img_output);
    img_background.copyTo(img_bgmodel);


    firstTime = false;
}

void MultiCues::init(const cv::Mat &img_input, cv::Mat &img_outfg, cv::Mat &img_outbg) {
    assert(img_input.empty() == false);
    img_outfg = cv::Mat::zeros(img_input.size(), CV_8UC1);
    img_outbg = cv::Mat::zeros(img_input.size(), CV_8UC1);
}

void MultiCues::save_config(cv::FileStorage &fs) {
    fs << "g_fLearningRate" << g_fLearningRate;
    fs << "g_iAbsortionPeriod" << g_iAbsortionPeriod;
    fs << "g_iC_ModelThreshold" << g_iC_ModelThreshold;
    fs << "g_iT_ModelThreshold" << g_iT_ModelThreshold;
    fs << "g_iBackClearPeriod" << g_iBackClearPeriod;
    fs << "g_iCacheClearPeriod" << g_iCacheClearPeriod;
    fs << "g_nNeighborNum" << g_nNeighborNum;
    fs << "g_nRadius" << g_nRadius;
    fs << "g_nTextureTrainVolRange" << g_nTextureTrainVolRange;
    fs << "g_bAbsorptionEnable" << g_bAbsorptionEnable;
    fs << "g_iTrainingPeriod" << g_iTrainingPeriod;
    fs << "g_iRWidth" << g_iRWidth;
    fs << "g_iRHeight" << g_iRHeight;
    fs << "showOutput" << showOutput;
}

void MultiCues::load_config(cv::FileStorage &fs) {
    fs["g_fLearningRate"] >> g_fLearningRate;
    fs["g_iAbsortionPeriod"] >> g_iAbsortionPeriod;
    fs["g_iC_ModelThreshold"] >> g_iC_ModelThreshold;
    fs["g_iT_ModelThreshold"] >> g_iT_ModelThreshold;
    fs["g_iBackClearPeriod"] >> g_iBackClearPeriod;
    fs["g_iCacheClearPeriod"] >> g_iCacheClearPeriod;
    fs["g_nNeighborNum"] >> g_nNeighborNum;
    fs["g_nRadius"] >> g_nRadius;
    fs["g_nTextureTrainVolRange"] >> g_nTextureTrainVolRange;
    fs["g_bAbsorptionEnable"] >> g_bAbsorptionEnable;
    fs["g_iTrainingPeriod"] >> g_iTrainingPeriod;
    fs["g_iRWidth"] >> g_iRWidth;
    fs["g_iRHeight"] >> g_iRHeight;
    fs["showOutput"] >> showOutput;
}

//-----------------------------------------------------------------------------------------------------------------------------------------//
//													the system initialization function													   //
//-----------------------------------------------------------------------------------------------------------------------------------------//
void MultiCues::Initialize(IplImage* frame)
{
    int i, j;

    g_iHeight = frame->height;
    g_iWidth = frame->width;

    Destroy();

    //--------------------------------------------------------
    // memory initialization
    //--------------------------------------------------------
    g_ResizedFrame = cvCreateImage(cvSize(g_iRWidth, g_iRHeight), IPL_DEPTH_8U, 1);

    g_aGaussFilteredFrame = (uchar***)malloc(sizeof(uchar**)*g_iRHeight);
    for (i = 0; i < g_iRHeight; i++) {
        g_aGaussFilteredFrame[i] = (uchar**)malloc(sizeof(uchar*)*g_iRWidth);
        for (j = 0; j < g_iRWidth; j++) g_aGaussFilteredFrame[i][j] = (uchar*)malloc(sizeof(uchar));
    }

    g_aXYZFrame = (uchar***)malloc(sizeof(uchar**)*g_iRHeight);
    for (i = 0; i < g_iRHeight; i++) {
        g_aXYZFrame[i] = (uchar**)malloc(sizeof(uchar*)*g_iRWidth);
        for (j = 0; j < g_iRWidth; j++) g_aXYZFrame[i][j] = (uchar*)malloc(sizeof(uchar) * 3);
    }

    g_aLandmarkArray = (uchar**)malloc(sizeof(uchar*)*g_iRHeight);
    for (i = 0; i < g_iRHeight; i++) g_aLandmarkArray[i] = (uchar*)malloc(sizeof(uchar)*g_iRWidth);

    g_aResizedForeMap = (uchar**)malloc(sizeof(uchar*)*g_iRHeight);
    for (i = 0; i < g_iRHeight; i++) g_aResizedForeMap[i] = (uchar*)malloc(sizeof(uchar)*g_iRWidth);

    g_aForegroundMap = (uchar**)malloc(sizeof(uchar*)*g_iHeight);
    for (i = 0; i < g_iHeight; i++) g_aForegroundMap[i] = (uchar*)malloc(sizeof(uchar)*g_iWidth);

    g_aUpdateMap = (BOOL**)malloc(sizeof(BOOL*)*g_iRHeight);
    for (i = 0; i < g_iRHeight; i++) g_aUpdateMap[i] = (BOOL*)malloc(sizeof(BOOL)*g_iRWidth);

    // Huong add
    h_GrayFrame = (uchar**)malloc(sizeof(uchar**)*g_iRHeight);
    for (i = 0; i <g_iRHeight; i++) h_GrayFrame[i] = (uchar*)malloc(sizeof(uchar)*g_iRWidth);
    h_DiffFrame = cvCreateImage(cvSize(g_iRWidth, g_iRHeight), IPL_DEPTH_8U, 1);
    h_mGrayFrame = cvCreateImage(cvSize(g_iRWidth, g_iRHeight), IPL_DEPTH_8U, 1);


    //    h_DiffFrame = (uchar**)malloc(sizeof(uchar**)*g_iRHeight);
    //    for (i = 0; i<g_iRHeight; i++) h_mGrayFrame[i] = (uchar*)malloc(sizeof(uchar)*g_iRWidth);

    //Bound Box Related..
    int iElementNum = 300;
    g_BoundBoxInfo = (BoundingBoxInfo*)malloc(sizeof(BoundingBoxInfo));
    g_BoundBoxInfo->m_iArraySize = iElementNum;
    g_BoundBoxInfo->m_iBoundBoxNum = iElementNum;

    g_BoundBoxInfo->m_aLeft = (short*)malloc(sizeof(short)* iElementNum); g_BoundBoxInfo->m_aRight = (short*)malloc(sizeof(short)* iElementNum);
    g_BoundBoxInfo->m_aBottom = (short*)malloc(sizeof(short)* iElementNum); g_BoundBoxInfo->m_aUpper = (short*)malloc(sizeof(short)* iElementNum);

    g_BoundBoxInfo->m_aRLeft = (short*)malloc(sizeof(short)* iElementNum); g_BoundBoxInfo->m_aRRight = (short*)malloc(sizeof(short)* iElementNum);
    g_BoundBoxInfo->m_aRBottom = (short*)malloc(sizeof(short)* iElementNum); g_BoundBoxInfo->m_aRUpper = (short*)malloc(sizeof(short)* iElementNum);

    g_BoundBoxInfo->m_ValidBox = (BOOL*)malloc(sizeof(BOOL)* iElementNum);

    //--------------------------------------------------------
    // texture model related
    //--------------------------------------------------------
    T_AllocateTextureModelRelatedMemory();

    //--------------------------------------------------------
    // color moddel related
    //--------------------------------------------------------
    C_AllocateColorModelRelatedMemory();

    //--------------------------------------------------------
    // Gauss moddel related
    //--------------------------------------------------------
#ifdef GAUSS
    h_AllocateGaussModelRelatedMemory();
#endif

    g_bModelMemAllocated = true;
    g_bNonModelMemAllocated = true;

}

//-----------------------------------------------------------------------------------------------------------------------------------------//
//												the function to release allocated memories											       //
//-----------------------------------------------------------------------------------------------------------------------------------------//
void MultiCues::Destroy()
{
    if (g_bModelMemAllocated == false && g_bNonModelMemAllocated == false) return;

    //short nNeighborNum = g_nNeighborNum;

    if (g_bModelMemAllocated == true) {
        T_ReleaseTextureModelRelatedMemory();
        C_ReleaseColorModelRelatedMemory();
        h_ReleaseGaussModelRelatedMemory();

        g_bModelMemAllocated = false;
    }

    if (g_bNonModelMemAllocated == true) {

        cvReleaseImage(&g_ResizedFrame);
        cvReleaseImage(&h_DiffFrame);
        cvReleaseImage(&h_mGrayFrame);

        for (int i = 0; i < g_iRHeight; i++) {
            for (int j = 0; j < g_iRWidth; j++) free(g_aGaussFilteredFrame[i][j]);
            free(g_aGaussFilteredFrame[i]);
        }
        free(g_aGaussFilteredFrame);

        for (int i = 0; i < g_iRHeight; i++) {
            for (int j = 0; j < g_iRWidth; j++) free(g_aXYZFrame[i][j]);
            free(g_aXYZFrame[i]);
        }
        free(g_aXYZFrame);

        for (int i = 0; i < g_iRHeight; i++) free(g_aLandmarkArray[i]);
        free(g_aLandmarkArray);

        for (int i = 0; i < g_iRHeight; i++) free(g_aResizedForeMap[i]);
        free(g_aResizedForeMap);

        for (int i = 0; i < g_iHeight; i++) free(g_aForegroundMap[i]);
        free(g_aForegroundMap);

        for (int i = 0; i < g_iRHeight; i++) free(g_aUpdateMap[i]);
        free(g_aUpdateMap);
        // Huong add
        for(int i = 0; i< g_iRHeight; i++)
            free(h_GrayFrame[i]);


        free(g_BoundBoxInfo->m_aLeft); free(g_BoundBoxInfo->m_aRight); free(g_BoundBoxInfo->m_aBottom); free(g_BoundBoxInfo->m_aUpper);
        free(g_BoundBoxInfo->m_aRLeft); free(g_BoundBoxInfo->m_aRRight); free(g_BoundBoxInfo->m_aRBottom); free(g_BoundBoxInfo->m_aRUpper);
        free(g_BoundBoxInfo->m_ValidBox);
        free(g_BoundBoxInfo);
        g_bNonModelMemAllocated = false;
    }
}
void MultiCues::h_AllocateGaussModelRelatedMemory()
{
    h_modelW  = g_iRWidth / h_blocksize;
    h_modelH  = g_iRHeight/ h_blocksize;
    h_BGModel = (GaussBG***)malloc(sizeof(GaussBG**)* h_modelH);
    for(int j = 0 ; j < h_modelH; j++)
    {
        h_BGModel[j] = (GaussBG**)malloc(sizeof(GaussBG*)* h_modelW);
        for(int i = 0; i < h_modelW; i++)
        {
            h_BGModel[j][i] = (GaussBG*)malloc(sizeof(GaussBG));
            h_BGModel[j][i]->h_mean = 0;
            h_BGModel[j][i]->h_var = 0;
        }
    }
}

void MultiCues::h_ReleaseGaussModelRelatedMemory()
{
    for(int j = 0 ; j < h_modelH; j++)
    {
        for(int i = 0; i < h_modelW; i++)
        {
            free(h_BGModel[i][j]);
            //            free(h_BGModel[i][j]->h_mean);
            //            free(h_BGModel[i][j]->h_var);
        }
        free(h_BGModel[j]);
    }
    free(h_BGModel);
}

//-----------------------------------------------------------------------------------------------------------------------------------------//
//														the preprocessing function		    											   //
//-----------------------------------------------------------------------------------------------------------------------------------------//
void MultiCues::PreProcessing(IplImage* frame) {
    //image resize
    std::cout << "Input img numChannel = " << frame->nChannels << std::endl;
    ReduceImageSize(frame, g_ResizedFrame);
    cv::Mat resizeImg(g_iRHeight, g_iRWidth, CV_8UC1);
    resizeImg = cvarrToMat(g_ResizedFrame);
//    std::cout << "Resizeimg = \n" << resizeImg << std::endl;
    //Gaussian filtering
    GaussianFiltering(g_ResizedFrame, g_aGaussFilteredFrame);
    h_RGB2GRAY_Par(g_ResizedFrame, h_GrayFrame);
//    printf("blurredImg =\n");

}

//-----------------------------------------------------------------------------------------------------------------------------------------//
//														the background modeling function												   //
//-----------------------------------------------------------------------------------------------------------------------------------------//
void MultiCues::BackgroundModeling_Par(IplImage* frame) {

    //initialization
    if (g_iFrameCount == 0)
        getExeTime(" --InitMem Time = ", Initialize(frame));

    //Step1: pre-processing
    getExeTime(" --Preprocessing Time = ", PreProcessing(frame));

    int iH_Start = g_nRadius, iH_end = g_iRHeight - g_nRadius;
    int iW_Start = g_nRadius, iW_end = g_iRWidth - g_nRadius;

    auto start = getMoment;
    float fLearningRate = g_fLearningRate * 4;
    //Step2: background modeling
    for (int i = iH_Start; i < iH_end; i++) {
        for (int j = iW_Start; j < iW_end; j++) {
            point center;
            center.m_nX = j;
            center.m_nY = i;
            T_ModelConstruction(g_nTextureTrainVolRange,fLearningRate, h_GrayFrame, center, g_aNeighborDirection[i][j], g_TextureModel[i][j]);
            C_CodebookConstruction(h_GrayFrame, center.m_nX , center.m_nY , g_nColorTrainVolRange, fLearningRate, g_ColorModel[i][j]);
        }
    }


#ifdef printTexture
    printf("============ Texture Information ====================\n");

    printf("Texture Model = \n");
    for(int y = 0; y < g_iRHeight; y++)
    {
        for(int x = 0; x < g_iRWidth; x++)
        {
            for(int k = 0; k < g_nNeighborNum; k++)
            {
                printf("%d\t", g_TextureModel[y][x][k]->m_iNumEntries);
            }
        }
        printf("\n");
    }
    printf("Texture Codeword Mean = \n");
    for(int y = 0; y < g_iRHeight; y++)
    {
        for(int x = 0; x < g_iRWidth; x++)
        {
            for(int k = 0; k < g_nNeighborNum; k++)
            {
                for(int j = 0; j < g_TextureModel[y][x][k]->m_iNumEntries; j++)
                {

                    printf("%f\t", g_TextureModel[y][x][k]->m_Codewords[j]->m_fMean);
                }
            }
        }
        printf("\n");
    }

//    printf("Texture Codeword Low Thresh = \n");
//    for(int y = 0; y < g_iRHeight; y++)
//    {
//        for(int x = 0; x < g_iRWidth; x++)
//        {
//            for(int k = 0; k < g_nNeighborNum; k++)
//            {
//                for(int j = 0; j < g_TextureModel[y][x][k]->m_iNumEntries; j++)
//                {

//                    printf("%f\t", g_TextureModel[y][x][k]->m_Codewords[j]->m_fLowThre);
//                }
//            }
//        }
//        printf("\n");
//    }

//    printf("Texture Codeword High Thresh = \n");
//    for(int y = 0; y < g_iRHeight; y++)
//    {
//        for(int x = 0; x < g_iRWidth; x++)
//        {
//            for(int k = 0; k < g_nNeighborNum; k++)
//            {
//                for(int j = 0; j < g_TextureModel[y][x][k]->m_iNumEntries; j++)
//                {

//                    printf("%f\t", g_TextureModel[y][x][k]->m_Codewords[j]->m_fHighThre);
//                }
//            }
//        }
//        printf("\n");
//    }

#endif
#ifdef printColor
    printf("============ Color Information ======================\n");

    printf("Color Model iNumEntri= \n");
    for(int y = 0; y < g_iRHeight; y++)
    {
        for(int x = 0; x < g_iRWidth; x++)
        {

            printf("%d\t", g_ColorModel[y][x]->m_iNumEntries);

        }
        printf("\n");
    }

    printf("Color Codeword Mean = \n");
    for(int y = 0; y < g_iRHeight; y++)
    {
        for(int x = 0; x < g_iRWidth; x++)
        {
            for(int k = 0; k < g_ColorModel[y][x]->m_iNumEntries; k++)
            {
                printf("%f\t", g_ColorModel[y][x]->m_Codewords[k]->m_dMean);
            }
        }
        printf("\n");
    }
#endif
    //Step3: Clear non-essential codewords
    if (g_iFrameCount == g_iTrainingPeriod) {
        auto start = getMoment;
        for (int i = 0; i < g_iRHeight; i++) {
            for (int j = 0; j < g_iRWidth; j++) {
                T_ClearNonEssentialEntries(g_iTrainingPeriod, g_TextureModel[i][j]);
                C_ClearNonEssentialEntries(g_iTrainingPeriod, g_ColorModel[i][j]);
            }
        }
        auto end = getMoment;
        getTimeElapsed("xxxxx Clear Time = ", end, start);
        //		g_iFrameCount++;

    }
    // Step4: Using Gauss Model
#ifdef GAUSS
    BOOL** aUpdateMap = (BOOL**)malloc(sizeof(BOOL*)*g_iRHeight);
    for (int i = 0; i < g_iRHeight; i++) {
        aUpdateMap[i] = (BOOL*)malloc(sizeof(BOOL)*g_iRWidth);
        for (int j = 0; j < g_iRWidth; j++) aUpdateMap[i][j] = true;
    }
    h_CreatGaussBG(h_BGModel, h_GrayFrame, 0.35, aUpdateMap);
#endif
    auto end = getMoment;
    getTimeElapsed(" -- Model Construction Time = ", end, start);
}
//-----------------------------------------------------------------------------------------------------------------------------------------//
//														the background GaussModel							                       //
//-----------------------------------------------------------------------------------------------------------------------------------------//
void MultiCues::h_CreatGaussBG(GaussBG*** gModel, uchar** aP, float h_learningrate, BOOL** updateMap)
{
    int h_gwidth = g_iRWidth;
    int h_ghight = g_iRHeight;
    //    gModel->h_mean = new float[gModel->h_modelW * gModel->h_modelH];

    for(int j = 0 ; j < h_modelH; j++)
    {
        for(int i = 0; i < h_modelW; i++)
        {
            float curmean = 0 ;
            float npixel = 0 ;
            for(int jj = 0 ; jj < h_blocksize; jj++)
            {
                for(int ii = 0; ii < h_blocksize; ii++)
                {
                    int idx = ii + i;
                    int idy = jj + j;
                    //                    cout <<" Value of pixel: " << (float)aP[idy][idx] << endl;
                    if(idx < 0 || idx >= h_gwidth || idy < 0 || idy >= h_ghight || updateMap[idy][idx] == false)
                    {
                        continue;
                    }
                    curmean += (float)aP[idy][idx];
                    npixel += 1;

                }
            }
            //            cout << " Curmean and number pixel " << curmean << " - " << npixel << endl;
            if(npixel != 0)
            {
                curmean = curmean / npixel;
            }
            else curmean = 0;

            gModel[j][i]->h_mean =(1-h_learningrate)* gModel[j][i]->h_mean +  h_learningrate * curmean ;
        }
    }
    for(int j = 0; j < h_modelH; j++){
        for(int i = 0; i < h_modelW ; i++)
        {
            float obs_var = gModel[j][i]->h_var ;
            for(int jj = 0 ; jj < h_blocksize; jj++){
                for (int ii = 0; ii < h_blocksize; ii++)
                {
                    int idx = i + ii;
                    int idy = j + jj;
                    if(idx < 0 || idx >= h_gwidth || idy < 0 || idy > h_ghight || updateMap[idy][idx] == false)
                    {
                        continue;
                    }

                    float fDiff = (float)aP[idy][idx] - gModel[j][i]->h_mean;
                    float pixelDist = pow(fDiff, (int)2);
                    obs_var = MAX(obs_var, pixelDist);
                }
            }
            gModel[j][i]->h_var = (1 - h_learningrate) * gModel[j][i]->h_var + h_learningrate * obs_var;
        }
    }
}
void MultiCues::h_UpdateGaussBG(IplImage* frame, GaussBG*** gModel, BoundingBoxInfo* BoundBoxInfo, uchar** gray, BOOL** updatemap, uchar** aResForeMap)
{

    int iBox_x, iBox_y, iBox_w, iBox_h, idxI, idxJ;
    cv::Mat disframe = cv::cvarrToMat(frame, true);

    if(BoundBoxInfo->m_iBoundBoxNum != 0 )
    {
        for( int i = 0; i < BoundBoxInfo->m_iBoundBoxNum; i++)
        {
            if(BoundBoxInfo->m_ValidBox[i] == true)
            {
                iBox_x = BoundBoxInfo->m_aRLeft[i];
                iBox_y = BoundBoxInfo->m_aRUpper[i];
                iBox_w = BoundBoxInfo->m_aRRight[i] - BoundBoxInfo->m_aRLeft[i];
                iBox_h = BoundBoxInfo->m_aRBottom[i] - BoundBoxInfo->m_aRUpper[i];

                int npixel = 0;
                int tpixel = 0;
                for (int y = iBox_y; y < iBox_y + iBox_h; y++)
                {
                    for (int x = iBox_x ; x < iBox_x+ iBox_w; x++)
                    {
                        // Topleft
                        idxI = floor(x/h_blocksize);
                        idxJ = floor(y/h_blocksize);
                        if(idxI > 0 && idxI < h_modelW & idxJ >0 && idxJ < h_modelH)
                        {

                            float fDiff = (float)gray[y][x] - gModel[idxJ][idxI]->h_mean;
                            float pixelDist = pow(fDiff, (int)2);
                            //                           cout << " **************fDiff *******************" << pixelDist << "**********" <<
                            //                                   THRES_BG_REFINE * gModel[idxJ][idxI]->h_var << endl;
                            if (pixelDist < THRES_BG_REFINE * gModel[idxJ][idxI]->h_var) // check it is background
                            {
                                //                                    updatemap[y][x] = true;
                                npixel++;

                            }
                        }
                        // Refine FG mask
                    }
                }
#ifdef DEBUG
                int w = BoundBoxInfo->m_aRight[i] - BoundBoxInfo->m_aLeft[i];
                int h = BoundBoxInfo->m_aBottom[i] - BoundBoxInfo->m_aUpper[i];
                int ix = BoundBoxInfo->m_aLeft[i];
                int iy = BoundBoxInfo->m_aUpper[i];
                cv::Rect rr = cv::Rect(ix, iy, w, h);
#endif

                if(float(npixel / (iBox_h * iBox_w)) > 0.1)

                {
                    BoundBoxInfo->m_ValidBox[i] = false;

                    for (int y = iBox_y; y < iBox_y + iBox_h; y++)
                    {
                        for (int x = iBox_x ; x < iBox_x+ iBox_w; x++)
                        {
                            if(aResForeMap[y][x] == 255)
                            {
                                updatemap[y][x] = true;
                            }

                        }
                    }
#ifdef DEBUG
                    cv::rectangle(disframe, rr, cv::Scalar(0, 255, 0), 2, 4);
#endif

                }
#ifdef DEBUG
                else
                {
                    cv::rectangle(disframe, rr, cv::Scalar(0, 0, 0), 2, 4);

                }
                string disfrname = DEBUGPATH + "/disframe.jpg";
                cv::imwrite(disfrname, disframe);
                cv::imshow("Disframe", disframe);
                cv::waitKey(0);
#endif
            } // end box True

        }// end i
    }
}
//-----------------------------------------------------------------------------------------------------------------------------------------//
//														the background subtraction function							                       //
//-----------------------------------------------------------------------------------------------------------------------------------------//
void MultiCues::ForegroundExtraction(IplImage* frame) {

    //Step1:pre-processing
    getExeTime(" ++ Preprocessing Time = ", PreProcessing(frame));

    //	Step3: texture-model based process

    // Huong thay doi
    getExeTime(" ++ Texture GetConfidenceMap Time = ",
               T_GetConfidenceMap_Par(h_GrayFrame, g_aTextureConfMap, g_aNeighborDirection, g_TextureModel));

//    printf("ConfMap = \n");
//    for(int y = 0; y < g_iRHeight; y++)
//    {
//        for(int x = 0; x < g_iRWidth; x++)
//        {
//            printf("%.3f\t", g_aTextureConfMap[y][x]);
//        }
//        printf("\n");
//    }
//    printf("Confidence Map = \n");
//    for(int y = 0; y < g_iRHeight; y++)
//    {
//        for(int x = 0; x < g_iRWidth; x++)
//        {
//            int idx = y * g_iRWidth + x;
//            printf("%.3f\t", g_aTextureConfMap[y][x]);
//        }
//        printf("\n");
//    }
    //Step2: color-model based verification

    getExeTime(" ++ CreateLandMark Time = ", CreateLandmarkArray_Par(g_fConfidenceThre, g_nColorTrainVolRange, g_aTextureConfMap, g_nNeighborNum, g_aXYZFrame, h_GrayFrame,g_aNeighborDirection,
                                                                     g_TextureModel, g_ColorModel, g_aLandmarkArray));

    //    cvShowImage("Landmark", *g_aLandmarkArray);
    //Step3: verification procedures
    getExeTime(" ++ PostProcessing Time = ", PostProcessing(frame));

}

//-----------------------------------------------------------------------------------------------------------------------------------------//
//														the post-processing function													   //
//-----------------------------------------------------------------------------------------------------------------------------------------//
void MultiCues::PostProcessing(IplImage* frame) {

    //Step1: morphological operation
    MorphologicalOpearions(g_aLandmarkArray, g_aResizedForeMap, 0.5, 5, g_iRWidth, g_iRHeight);
    g_bForegroundMapEnable = true;

    //Step2: labeling
    int** aLabelTable = (int**)malloc(sizeof(int*)*g_iRHeight);
    for (int i = 0; i < g_iRHeight; i++)  aLabelTable[i] = (int*)malloc(sizeof(int)*g_iRWidth);

    int iLabelCount;
    Labeling(g_aResizedForeMap, &iLabelCount, aLabelTable);


    //Step3: getting bounding boxes for each candidate fore-blob
    //	SetBoundingBox(iLabelCount, aLabelTable);

    //Step4: size  and appearance based verification
    BoundBoxVerification(frame, g_aResizedForeMap, g_BoundBoxInfo);

    //Step5: Foreground Region
    RemovingInvalidForeRegions(g_aResizedForeMap, g_BoundBoxInfo);

    //    h_UpdateGaussBG(h_BGModel, g_BoundBoxInfo, h_GrayFrame);

    for (int i = 0; i < g_iRHeight; i++) free(aLabelTable[i]);
    free(aLabelTable);
}

//-----------------------------------------------------------------------------------------------------------------------------------------//
//														the background-model update function			                                   //
//-----------------------------------------------------------------------------------------------------------------------------------------//
void MultiCues::UpdateModel_Par() {
    //short nNeighborNum = g_nNeighborNum;

    //Step1: update map construction
    for (int i = 0; i < g_iRHeight; i++) {
        for (int j = 0; j < g_iRWidth; j++) {
            g_aUpdateMap[i][j] = true;
        }
    }

    for (int k = 0; k < g_BoundBoxInfo->m_iBoundBoxNum; k++) {
        if (g_BoundBoxInfo->m_ValidBox[k] == true) {
            for (int i = g_BoundBoxInfo->m_aRUpper[k]; i <= g_BoundBoxInfo->m_aRBottom[k]; i++) {
                for (int j = g_BoundBoxInfo->m_aRLeft[k]; j <= g_BoundBoxInfo->m_aRRight[k]; j++) {
                    if(g_aResizedForeMap[i][j] == (uchar)255){
                        g_aUpdateMap[i][j] = false;}
                }
            }
        }
    }

    //Step2: update
    int iH_Start = g_nRadius, iH_End = g_iRHeight - g_nRadius;
    int iW_Start = g_nRadius, iW_End = g_iRWidth - g_nRadius;

    float fLearningRate = (float)g_fLearningRate;

    for (int i = iH_Start; i < iH_End; i++) {
        for (int j = iW_Start; j < iW_End; j++) {

            point center;
            center.m_nX = j;
            center.m_nY = i;

            if (g_aUpdateMap[i][j] == true) {
                //model update
                T_ModelConstruction(g_nTextureTrainVolRange, fLearningRate, h_GrayFrame, center, g_aNeighborDirection[i][j], g_TextureModel[i][j]);
                C_CodebookConstruction(h_GrayFrame, j, i, g_nColorTrainVolRange, fLearningRate, g_ColorModel[i][j]);

                //clearing non-essential codewords
                T_ClearNonEssentialEntries(g_iBackClearPeriod, g_TextureModel[i][j]);
                C_ClearNonEssentialEntries(g_iBackClearPeriod, g_ColorModel[i][j]);

            }
            else {
                if (g_bAbsorptionEnable == true) {
                    //model update
                    T_ModelConstruction(g_nTextureTrainVolRange, fLearningRate, h_GrayFrame, center, g_aNeighborDirection[i][j], g_TCacheBook[i][j]);
                    C_CodebookConstruction(h_GrayFrame, j, i, g_nColorTrainVolRange, fLearningRate, g_CCacheBook[i][j]);
                    //clearing non-essential codewords
                    T_Absorption(g_iAbsortionPeriod, center, g_aTContinuousCnt, g_aTReferredIndex, g_TextureModel[i][j], g_TCacheBook[i][j]);
                    C_Absorption(g_iAbsortionPeriod, center, g_aCContinuousCnt, g_aCReferredIndex, g_ColorModel[i][j], g_CCacheBook[i][j]);

                }
            }

            //clearing non-essential codewords for cache-books
            if (g_bAbsorptionEnable == true) {
                T_ClearNonEssentialEntriesForCachebook(g_aLandmarkArray[i][j], g_aTReferredIndex[i][j], 10, g_TCacheBook[i][j]);
                C_ClearNonEssentialEntriesForCachebook(g_aLandmarkArray[i][j], g_aCReferredIndex[i][j], 10, g_CCacheBook[i][j]);
            }
        }
    }
    // Update Guass Model
#ifdef GAUSS
    h_CreatGaussBG(h_BGModel,  h_GrayFrame, 0.05, g_aUpdateMap);
#endif

}

//-----------------------------------------------------------------------------------------------------------------------------------------//
//														Huong thay doi                      											   //
//-----------------------------------------------------------------------------------------------------------------------------------------//
void MultiCues::CreateLandmarkArray_Par(float fConfThre, short nTrainVolRange, float**aConfMap, int iNehborNum, uchar*** aXYZ,uchar** gray,
                                        point*** aNeiDir, TextureModel**** TModel, ColorModel*** CModel, uchar**aLandmarkArr) {

    int iBound_w = g_iRWidth - g_nRadius;
    int iBound_h = g_iRHeight - g_nRadius;
    IplImage* hLandmark;
    hLandmark = cvCreateImage(cvSize(g_iRWidth, g_iRHeight), IPL_DEPTH_8U, 1);
    for (int i = 0; i < g_iRHeight; i++) {
        for (int j = 0; j < g_iRWidth; j++) {
            if (i < g_nRadius || i >= iBound_h || j < g_nRadius || j >= iBound_w) {
                aLandmarkArr[i][j] = 0;
                hLandmark->imageData[j + i * g_iRWidth] = 0;
                continue;
            }

            double tmp = aConfMap[i][j];

            if (tmp > fConfThre) {aLandmarkArr[i][j] = 255;
                hLandmark->imageData[j + i * g_iRWidth] = 255; }
            else {
                aLandmarkArr[i][j] = 0;
                hLandmark->imageData[j + i * g_iRWidth] = 0;
                //Calculating texture amount in the background
                double dBackAmt, dCnt;
                dBackAmt = dCnt = 0;

                for (int m = 0; m < iNehborNum; m++) {
                    for (int n = 0; n < TModel[i][j][m]->m_iNumEntries; n++) {
                        dBackAmt += TModel[i][j][m]->m_Codewords[n]->m_fMean;
                        dCnt++;
                    }
                }
                dBackAmt /= dCnt;

                //Calculating texture amount in the input image
                double dTemp, dInputAmt = 0;
                for (int m = 0; m < iNehborNum; m++) {
                    dTemp = gray[i][j] - gray[aNeiDir[i][j][m].m_nY][aNeiDir[i][j][m].m_nX];

                    if (dTemp >= 0) dInputAmt += dTemp;
                    else dInputAmt -= dTemp;

                }

                //If there are only few textures in both background and input image
                if (dBackAmt < 50 && dInputAmt < 50) { // 50
                    //Conduct color codebook matching
                    BOOL bMatched = false;
                    for (int m = 0; m < CModel[i][j]->m_iNumEntries; m++) {

                        int iMatchedCount = 0;
                        double dLowThre = CModel[i][j]->m_Codewords[m]->m_dMean - nTrainVolRange - 15;
                        double dHighThre = CModel[i][j]->m_Codewords[m]->m_dMean + nTrainVolRange + 15;
                        if (dLowThre <= gray[i][j] && gray[i][j]<= dHighThre)
                            iMatchedCount++;


                        if (iMatchedCount == 1) {
                            bMatched = true;
                            break;
                        }

                    }
                    if (bMatched == true) {aLandmarkArr[i][j] = 0;
                        hLandmark->imageData[j + i*g_iRWidth] = 0; }
                    else{ aLandmarkArr[i][j] = 255;
                        hLandmark->imageData[j+i * g_iRWidth] = 255;  }

                }

            }
        }
    }
    cvShowImage("landmark", hLandmark);
#ifdef DEBUG
    cv::Mat rfcolor = cvarrToMat(hLandmark, true);
    string rfcolorResponse =  DEBUGPATH + "/rfColorMap.jpg";
    cv::imwrite(rfcolorResponse, rfcolor);
#endif
}

//-----------------------------------------------------------------------------------------------------------------------------------------//
//													the Gaussian filtering function								                           //
//-----------------------------------------------------------------------------------------------------------------------------------------//
void MultiCues::GaussianFiltering(IplImage* frame, uchar*** aFilteredFrame) {

    double dSigma = 0.7;

    if (dSigma == 0) {
        for (int i = 0; i < g_iRHeight; i++) {
            for (int j = 0; j < g_iRWidth; j++) {
                aFilteredFrame[i][j][0] = frame->imageData[i*frame->widthStep + j * 3];
                aFilteredFrame[i][j][1] = frame->imageData[i*frame->widthStep + j * 3 + 1];
                aFilteredFrame[i][j][2] = frame->imageData[i*frame->widthStep + j * 3 + 2];
            }
        }
    }
    else
    {
        cv::Mat temp_img = cv::cvarrToMat(frame, true);
        cv::GaussianBlur(temp_img, temp_img, cv::Size(7, 7), dSigma);
        cv::imshow("blur", temp_img);
        //Store results into aFilteredFrame[][][]
        //IplImage* img = &IplImage(temp_img);
        IplImage* img = new IplImage(temp_img);
        //int iWidthStep = img->widthStep;

        for (int i = 0; i < g_iRHeight; i++) {
            for (int j = 0; j < g_iRWidth; j++) {
                aFilteredFrame[i][j][0] = img->imageData[i*img->widthStep + j * 3];
                aFilteredFrame[i][j][1] = img->imageData[i*img->widthStep + j * 3 + 1];
                aFilteredFrame[i][j][2] = img->imageData[i*img->widthStep + j * 3 + 2];
            }
        }
        delete img;
    }
}
void MultiCues::h_RGB2GRAY_Par(IplImage* frame, uchar **gray)
{
    cv::Mat temp_img = cv::cvarrToMat(frame, true);
//    cv::imshow("resizeImg", temp_img);
//    std::cout << "Resize Img = \n" << temp_img << std::endl;
//    cv::GaussianBlur(temp_img, temp_img, cv::Size(7, 7), 0.7);

    cv::cuda::GpuMat resizeWarp(temp_img.size(), CV_8UC1);
    cv::cuda::GpuMat blurWarp(temp_img.size(), CV_8UC1);
    resizeWarp.upload(temp_img);
    cv::Ptr<cv::cuda::Filter> filter = cv::cuda::createGaussianFilter(CV_8UC1, CV_8UC1, cv::Size(7,7),
                                                                      0.7);
    filter->apply(resizeWarp, blurWarp);
    blurWarp.download(temp_img);

//    std::cout << "BlurImg = \n" << temp_img << std::endl;
    //       IplImage* tempt = cvCreateImage(cvGetSize(frame), IPL_DEPTH_8U, 1);
;
    IplImage* tempt = new IplImage(temp_img);
    for(int i = 0; i < g_iRHeight; i++)
    {
        for(int j = 0; j < g_iRWidth; j++)
        {
            gray[i][j] = (uchar) tempt->imageData[j+ i * tempt->widthStep]; //hsv[i][j][2];
        }
    }
    delete tempt;
}
//------------------------------------------------------------------------------------------------------------------------------------//
//														the image resize function									                 //
//------------------------------------------------------------------------------------------------------------------------------------//
void MultiCues::ReduceImageSize(IplImage* SrcImage, IplImage* DstImage) {

    int iChannel = 1;

    double dResizeFactor_w = (double)g_iWidth / (double)g_iRWidth;
    double dResizeFactor_h = (double)g_iHeight / (double)g_iRHeight;

    for (int i = 0; i < g_iRHeight; i++) {
        for (int j = 0; j < g_iRWidth; j++) {
            int iSrcY = (int)(i*dResizeFactor_h);
            int iSrcX = (int)(j*dResizeFactor_w);
            DstImage->imageData[i * g_iRWidth + j] = SrcImage->imageData[iSrcY * g_iWidth + iSrcX];
//            for (int k = 0; k < iChannel; k++) DstImage->imageData[i*DstImage->widthStep + j * 3 + k]
//                    = SrcImage->imageData[iSrcY*SrcImage->widthStep + iSrcX * 3 + k];
        }
    }
//    cvShowImage("resize", DstImage);
}

//------------------------------------------------------------------------------------------------------------------------------------//
//											    the color space conversion function                                                   //
//------------------------------------------------------------------------------------------------------------------------------------//
void MultiCues::BGR2HSVxyz_Par(uchar*** aBGR, uchar*** aXYZ) {

    double dH_ratio = (2 * PI) / 360;

    for (int i = 0; i < g_iRHeight; i++) {

        double dR, dG, dB;
        double dMax, dMin;

        double dH, dS, dV;

        for (int j = 0; j < g_iRWidth; j++) {

            dB = (double)(aBGR[i][j][0]) / 255;
            dG = (double)(aBGR[i][j][1]) / 255;
            dR = (double)(aBGR[i][j][2]) / 255;


            //Find max, min
            dMin = MIN3(dR, dG, dB);
            dMax = MAX3(dR, dG, dB);


            //Get V
            dV = dMax;

            //Get S, H
            if (dV == 0) dS = dH = 0;
            else {

                //S value
                dS = (dMax - dMin) / dMax;

                if (dS == 0) dH = 0;
                else {
                    //H value
                    if (dMax == dR) {
                        dH = 60 * (dG - dB) / dS;
                        if (dH < 0) dH = 360 + dH;
                    }
                    else if (dMax == dG) dH = 120 + 60 * (dB - dR) / dS;
                    else dH = 240 + 60 * (dR - dG) / dS;
                }
            }
            dH = dH * dH_ratio;

            aXYZ[i][j][0] = (uchar)0; //((dV * dS * cos(dH) * 127.5) + 127.5);		//X  --> 0~255
            aXYZ[i][j][1] = (uchar)0; // ((dV * dS * sin(dH) * 127.5) + 127.5);		//Y  --> 0~255
            aXYZ[i][j][2] = (uchar)(dV * 255);							    //Z  --> 0~255
            std::cout << (int) aXYZ[i][j][0] << " - " << (int)aXYZ[i][j][1] << " - " << (int)aXYZ[i][j][2] << std::endl;
        }
    }
}

//-----------------------------------------------------------------------------------------------------------------------------------------//
//												the function to get enlarged confidence map												   //
//-----------------------------------------------------------------------------------------------------------------------------------------//
void MultiCues::GetEnlargedMap(float** aOriginMap, float** aEnlargedMap) {
    int i, j;

    short nSrcX;
    short nSrcY;

    double dEWweight, dNSweight;
    double dEWtop, dEWbottom;

    double dNW; //north-west
    double dNE; //north-east
    double dSW; //south-west
    double dSE; //south-east

    double dScaleFactor_w = ((double)g_iWidth) / ((double)g_iRWidth);
    double dScaleFactor_h = ((double)g_iHeight) / ((double)g_iRHeight);

    for (i = 0; i < g_iHeight; i++) {
        for (j = 0; j < g_iWidth; j++) {
            //backward mapping
            nSrcY = (int)(i / dScaleFactor_h);
            nSrcX = (int)(j / dScaleFactor_w);

            if (nSrcY == (g_iRHeight - 1)) nSrcY -= 1;
            if (nSrcX == (g_iRWidth - 1)) nSrcX -= 1;

            dEWweight = i / dScaleFactor_h - nSrcY;
            dNSweight = j / dScaleFactor_w - nSrcX;

            dNW = (double)aOriginMap[nSrcY][nSrcX];
            dNE = (double)aOriginMap[nSrcY][nSrcX + 1];
            dSW = (double)aOriginMap[nSrcY + 1][nSrcX];
            dSE = (double)aOriginMap[nSrcY + 1][nSrcX + 1];

            // interpolation
            dEWtop = dNW + dEWweight * (dNE - dNW);
            dEWbottom = dSW + dEWweight * (dSE - dSW);

            aEnlargedMap[i][j] = (float)(dEWtop + dNSweight * (dEWbottom - dEWtop));
        }
    }
}

//-----------------------------------------------------------------------------------------------------------------------------------------//
//										   				 the morphological operation function				    				   		   //
//-----------------------------------------------------------------------------------------------------------------------------------------//
void MultiCues::MorphologicalOpearions(uchar** aInput, uchar** aOutput, double dThresholdRatio, int iMaskSize, int iWidth, int iHeight) {
#ifdef OPEN_CV
    IplImage* mInput;
    mInput = cvCreateImage(cvSize(iWidth, iHeight), IPL_DEPTH_8U, 1);
    for(int i = 0; i< iHeight ; i++)
    {
        for(int j = 0; j < iWidth; j++)
        {
            mInput->imageData[j + i * iWidth] = aInput[i][j];
        }
    }
    IplImage* dst;
    dst = cvCreateImage(cvSize(iWidth, iHeight), IPL_DEPTH_8U, 1);
    IplConvKernel* convKernel = cvCreateStructuringElementEx(3, 3, 1, 1, CV_SHAPE_ELLIPSE);
    //    for (int i = 0; i < 3; i++)
    //    {

    //       cvMorphologyEx(mInput, dst, NULL, convKernel, CV_MOP_OPEN);
    //       cvMorphologyEx(dst, dst, NULL, convKernel, CV_MOP_CLOSE);

    //    }
    cvDilate(mInput, dst, convKernel, 1);
    cv::Mat dila = cvarrToMat(dst, true);
    cvErode(dst, dst, convKernel, 1);

    cv::Mat erod = cvarrToMat(dst, true);

    cvAnd(dst, dst, dst, NULL);
    cv::Mat andt = cvarrToMat(dst, true);

    for(int i = 0; i < iHeight; i++)
    {
        for(int j = 0; j < iWidth; j++)
        {
            aOutput[i][j] = dst->imageData[j + i * iWidth];
        }
    }
#ifdef DEBUG
    Mat dstim = cvarrToMat(dst, true);
    string morphologyname = DEBUGPATH + "/morph.jpg";
    imwrite(morphologyname, dstim);
    string dilaname = DEBUGPATH + "/dilaname.jpg";
    string erodname = DEBUGPATH + "/erodname.jpg";
    string andname = DEBUGPATH + "/andname.jpg";

    imwrite(dilaname, dila);
    imwrite(erodname, erod);
    imwrite(andname, andt);

    imshow("dst", dstim);
    waitKey(0);
#endif
    cvReleaseImage(&mInput);
    cvReleaseImage(&dst);

    //    cvRelease(&convKernel);


    //#else
    int iOffset = (int)(iMaskSize / 2);

    int iBound_w = iWidth - iOffset;
    int iBound_h = iHeight - iOffset;

    uchar** aTemp = (uchar**)malloc(sizeof(uchar*)*iHeight);
    for (int i = 0; i < iHeight; i++) {
        aTemp[i] = (uchar*)malloc(sizeof(uchar)*iWidth);
    }

    for (int i = 0; i < iHeight; i++) {
        for (int j = 0; j < iWidth; j++) {
            aTemp[i][j] = aInput[i][j];
        }
    }

    int iThreshold = (int)(iMaskSize*iMaskSize*dThresholdRatio);
    for (int i = 0; i < iHeight; i++) {
        for (int j = 0; j < iWidth; j++) {

            if (i < iOffset || i >= iBound_h || j < iOffset || j >= iBound_w) {
                aOutput[i][j] = 0;
                //                hLandmark->imageData[j + i * iWidth] = 0;
                continue;
            }

            int iCnt = 0;
            for (int m = -iOffset; m <= iOffset; m++) {
                for (int n = -iOffset; n <= iOffset; n++) {
                    if (aTemp[i + m][j + n] == 255) iCnt++;
                }
            }

            if (iCnt >= iThreshold)
            {aOutput[i][j] = 255;
                //            hLandmark->imageData[j + i * iWidth] = 255;
            }
            else
            {aOutput[i][j] = 0;
                //            hLandmark->imageData[j + i * iWidth] = 0;
            }
        }
    }


    for (int i = 0; i < iHeight; i++) {
        free(aTemp[i]);
    }
    free(aTemp);
    IplImage* bLandmark;
    bLandmark = cvCreateImage(cvSize(iWidth, iHeight), IPL_DEPTH_8U, 1);
    IplImage* hLandmark;
    hLandmark = cvCreateImage(cvSize(iWidth, iHeight), IPL_DEPTH_8U, 1);
    for(int i = 0; i < iHeight; i++)
    {
        for(int j = 0; j < iWidth; j++)
        {
            hLandmark->imageData[j + i * iWidth] = aOutput[i][j];
            bLandmark->imageData[j + i * iWidth] = aInput[i][j];
        }
    }
    cv::Mat morphoimg = cv::cvarrToMat(hLandmark, true);
    cv::Mat input = cv::cvarrToMat(bLandmark, true);
    string oldmorphoname = DEBUGPATH + "/oldmorp.jpg";
    imwrite(oldmorphoname, morphoimg);
    //#ifdef DE
    //    cv::imshow("foremap", input);
    //    cv::imshow("Morphologyimage", morphoimg);
    //    cv::waitKey(0);
    cvReleaseImage(&hLandmark);
#endif
}
//-----------------------------------------------------------------------------------------------------------------------------------------//
//													2-raster scan pass based labeling function											   //
//-----------------------------------------------------------------------------------------------------------------------------------------//
void MultiCues::Labeling(uchar** aBinaryArray, int* pLabelCount, int** aLabelTable) {
#ifdef OPEN_CV

    cv::Mat mInput(cv::Size(g_iRWidth, g_iRHeight), CV_8UC1, 1);
    for(int i = 0; i< g_iRHeight ; i++)
    {
        for(int j = 0; j < g_iRWidth; j++)
        {
            mInput.at<uchar>(i,j) = aBinaryArray[i][j];
        }
    }

    cv::Mat stats, centroid, labelimg;
    int nLabels = cv::connectedComponentsWithStats(mInput, labelimg, stats, centroid, 4, CV_32S);
    g_BoundBoxInfo->m_iBoundBoxNum = nLabels -1;
    for(int i = 1 ; i<nLabels -1 ; i++)
    {
        g_BoundBoxInfo->m_aRLeft[i] = stats.at<int>(i, 0);
        g_BoundBoxInfo->m_aRRight[i] = stats.at<int>(i, 0) + stats.at<int>(i, 2);
        g_BoundBoxInfo->m_aRUpper[i] = stats.at<int>(i, 1);
        g_BoundBoxInfo->m_aRBottom[i] = stats.at<int>(i, 1) + stats.at<int>(i, 3);
        g_BoundBoxInfo->m_ValidBox[i] = true;
    }

    double dH_ratio = (double)g_iHeight / (double)g_iRHeight;
    double dW_ratio = (double)g_iWidth / (double)g_iRWidth;

    for (int i = 0; i < g_BoundBoxInfo->m_iBoundBoxNum; i++) {
        g_BoundBoxInfo->m_aLeft[i] = (int)(g_BoundBoxInfo->m_aRLeft[i] * dW_ratio);
        g_BoundBoxInfo->m_aUpper[i] = (int)(g_BoundBoxInfo->m_aRUpper[i] * dH_ratio);
        g_BoundBoxInfo->m_aRight[i] = (int)(g_BoundBoxInfo->m_aRRight[i] * dW_ratio);
        g_BoundBoxInfo->m_aBottom[i] = (int)(g_BoundBoxInfo->m_aRBottom[i] * dH_ratio);
    }

#else
    int x, y, i;		// pass 1,2
    int cnt = 0;		// pass 1
    int label = 0;	// pass 2

    int iSize = g_iRWidth * g_iRHeight;
    int iTableSize = iSize / 2;

    // initialize , table1 table1
    int* aPass1 = (int*)malloc(iSize * sizeof(int));
    int* aTable1 = (int*)malloc(iSize / 2 * sizeof(int));
    int* aTable2 = (int*)malloc(iSize / 2 * sizeof(int));

    memset(aPass1, 0, (iSize) * sizeof(int));
    for (y = 1; y < (g_iRHeight); y++) {
        for (x = 1; x < (g_iRWidth); x++) {
            aLabelTable[y][x] = 0;
        }
    }

    for (i = 0; i < iTableSize; i++) {
        aTable1[i] = i;
    }
    memset(aTable2, 0, iTableSize * sizeof(int));

    // pass 1
    for (y = 1; y < (g_iRHeight); y++) {
        for (x = 1; x < (g_iRWidth); x++) {

            if (aBinaryArray[y][x] == 255) { // fore ground??
                int up, le;

                up = aPass1[(y - 1)*(g_iRWidth)+(x)]; // up  index
                le = aPass1[(y)*(g_iRWidth)+(x - 1)]; // left index

                // case
                if (up == 0 && le == 0) {
                    ++cnt;
                    aPass1[y * g_iRWidth + x] = cnt;

                }
                else if (up != 0 && le != 0) {
                    if (up > le) {
                        aPass1[y *g_iRWidth + x] = le;
                        aTable1[up] = aTable1[le]; // update table1 table1
                    }
                    else {
                        aPass1[y * g_iRWidth + x] = up;
                        aTable1[le] = aTable1[up]; // update table1 table1
                    }
                }
                else {
                    aPass1[y * g_iRWidth + x] = up + le;
                }

            }

        }
    }

    // pass 2
    for (y = 1; y < (g_iRHeight); y++) {
        for (x = 1; x < (g_iRWidth); x++) {

            if (aPass1[y * g_iRWidth + x]) {
                int v = aTable1[aPass1[y * g_iRWidth + x]];

                if (aTable2[v] == 0) {
                    ++label;
                    aTable2[v] = label;
                }

                aLabelTable[y][x] = aTable2[v];
            }
        }
    }

    *pLabelCount = label;

    free(aPass1);
    free(aTable1);
    free(aTable2);
#endif
}

//-----------------------------------------------------------------------------------------------------------------------------------------//
//									the function to set bounding boxes for each candidate foreground regions					    	   //																									   //
//-----------------------------------------------------------------------------------------------------------------------------------------//
void MultiCues::SetBoundingBox(int iLabelCount, int** aLabelTable) {
    int iBoundBoxIndex;

    g_BoundBoxInfo->m_iBoundBoxNum = iLabelCount;

    for (int i = 0; i < g_BoundBoxInfo->m_iBoundBoxNum; i++) {
        g_BoundBoxInfo->m_aRLeft[i] = 9999;		//left
        g_BoundBoxInfo->m_aRUpper[i] = 9999;	//top
        g_BoundBoxInfo->m_aRRight[i] = 0;		//right
        g_BoundBoxInfo->m_aRBottom[i] = 0;		//bottom
    }

    //Step1: Set tight bounding boxes
    for (int i = 1; i < g_iRHeight; i++) {
        for (int j = 1; j < g_iRWidth; j++) {

            if (aLabelTable[i][j] == 0) continue;

            iBoundBoxIndex = aLabelTable[i][j] - 1;

            if (g_BoundBoxInfo->m_aRLeft[iBoundBoxIndex] > j) g_BoundBoxInfo->m_aRLeft[iBoundBoxIndex] = j;		//left
            if (g_BoundBoxInfo->m_aRUpper[iBoundBoxIndex] > i) g_BoundBoxInfo->m_aRUpper[iBoundBoxIndex] = i;		//top
            if (g_BoundBoxInfo->m_aRRight[iBoundBoxIndex] < j) g_BoundBoxInfo->m_aRRight[iBoundBoxIndex] = j;		//right
            if (g_BoundBoxInfo->m_aRBottom[iBoundBoxIndex] < i) g_BoundBoxInfo->m_aRBottom[iBoundBoxIndex] = i;	//bottom

        }
    }

    //Step2: Add margins.
    int iBoundary_w = (int)(g_iRWidth / 80), iBoundary_h = (int)(g_iRHeight / 60);

    for (int i = 0; i < g_BoundBoxInfo->m_iBoundBoxNum; i++) {

        g_BoundBoxInfo->m_aRLeft[i] -= iBoundary_w;
        if (g_BoundBoxInfo->m_aRLeft[i] < g_nRadius) g_BoundBoxInfo->m_aRLeft[i] = g_nRadius;									//left

        g_BoundBoxInfo->m_aRRight[i] += iBoundary_w;
        if (g_BoundBoxInfo->m_aRRight[i] >= g_iRWidth - g_nRadius) g_BoundBoxInfo->m_aRRight[i] = g_iRWidth - g_nRadius - 1;		    //Right

        g_BoundBoxInfo->m_aRUpper[i] -= iBoundary_h;
        if (g_BoundBoxInfo->m_aRUpper[i] < g_nRadius) g_BoundBoxInfo->m_aRUpper[i] = g_nRadius;									//Top

        g_BoundBoxInfo->m_aRBottom[i] += iBoundary_h;
        if (g_BoundBoxInfo->m_aRBottom[i] >= g_iRHeight - g_nRadius) g_BoundBoxInfo->m_aRBottom[i] = g_iRHeight - g_nRadius - 1;
    }

    double dH_ratio = (double)g_iHeight / (double)g_iRHeight;
    double dW_ratio = (double)g_iWidth / (double)g_iRWidth;

    for (int i = 0; i < g_BoundBoxInfo->m_iBoundBoxNum; i++) {
        g_BoundBoxInfo->m_aLeft[i] = (int)(g_BoundBoxInfo->m_aRLeft[i] * dW_ratio);
        g_BoundBoxInfo->m_aUpper[i] = (int)(g_BoundBoxInfo->m_aRUpper[i] * dH_ratio);
        g_BoundBoxInfo->m_aRight[i] = (int)(g_BoundBoxInfo->m_aRRight[i] * dW_ratio);
        g_BoundBoxInfo->m_aBottom[i] = (int)(g_BoundBoxInfo->m_aRBottom[i] * dH_ratio);
    }

}

//-----------------------------------------------------------------------------------------------------------------------------------------//
//														the box verification function													   //																							   //
//-----------------------------------------------------------------------------------------------------------------------------------------//
void MultiCues::BoundBoxVerification(IplImage* frame, uchar** aResForeMap, BoundingBoxInfo* BoundBoxInfo) {

    //Step1: Verification by the bounding box size
    EvaluateBoxSize(BoundBoxInfo);

    //Step1.5 : Verification by the bounding box size
    h_CaclOverlap(BoundBoxInfo);

    //Step2: Verification by checking whether the region is ghost
    EvaluateGhostRegion(frame, aResForeMap, BoundBoxInfo);

    //Step3: Counting the # of valid box
    g_iForegroundNum = 0;
    for (int i = 0; i < BoundBoxInfo->m_iBoundBoxNum; i++) {
        if (BoundBoxInfo->m_ValidBox[i] == true) g_iForegroundNum++;
    }
}

//-----------------------------------------------------------------------------------------------------------------------------------------//
//																the size based verification												   //																							   //
//-----------------------------------------------------------------------------------------------------------------------------------------//
void MultiCues::EvaluateBoxSize(BoundingBoxInfo* BoundBoxInfo) {

    //Set thresholds
    int iLowThreshold_w, iHighThreshold_w;
    iLowThreshold_w = g_iRWidth / 32; if (iLowThreshold_w < 5) iLowThreshold_w = 5;
    iHighThreshold_w = g_iRWidth - 5 ;

    int iLowThreshold_h, iHighThreshold_h;
    iLowThreshold_h = g_iRHeight / 24; if (iLowThreshold_h < 5) iLowThreshold_h = 5;
    iHighThreshold_h = g_iRHeight - 5;
    int iLowThreshold_area = iLowThreshold_h * iLowThreshold_w;
    int iHightThreshold_area = iHighThreshold_h * iHighThreshold_w;
    float iLowThreshold_rt = 0.2;
    float iHighThreshold_rt = 5.0;

    int iBoxWidth, iBoxHeight;

    //Perform verification.
    for (int i = 0; i < BoundBoxInfo->m_iBoundBoxNum; i++) {

        iBoxWidth = BoundBoxInfo->m_aRRight[i] - BoundBoxInfo->m_aRLeft[i];
        iBoxHeight = BoundBoxInfo->m_aRBottom[i] - BoundBoxInfo->m_aRUpper[i];
        int iBoxArea = iBoxHeight * iBoxWidth;

        float iBoxRt = (float) iBoxWidth / (float) iBoxHeight;
        //        cout << " RR " << iBoxRt << " - " << iBoxArea << " - " << iLowThreshold_area << " - " << i
        if (iLowThreshold_w <= iBoxWidth && iBoxWidth <= iHighThreshold_w &&
                iLowThreshold_h <= iBoxHeight && iBoxHeight <= iHighThreshold_h &&
                iLowThreshold_area <= iBoxArea && iBoxArea <= iHightThreshold_area &&
                iLowThreshold_rt <= iBoxRt  && iBoxRt <= iHighThreshold_rt) {
            BoundBoxInfo->m_ValidBox[i] = true;
        }
        else BoundBoxInfo->m_ValidBox[i] = false;
    }
}

//------------------------------------------------------------------------------------------------------------------------------------//
//														overlapped region removal													  //
//------------------------------------------------------------------------------------------------------------------------------------//
void MultiCues::EvaluateOverlapRegionSize(BoundingBoxInfo* SrcBoxInfo) {

    BOOL *aValidBoxFlag = new BOOL[SrcBoxInfo->m_iBoundBoxNum];
    for (int i = 0; i < SrcBoxInfo->m_iBoundBoxNum; i++) aValidBoxFlag[i] = true;

    int size1, size2;
    short *aLeft = SrcBoxInfo->m_aRLeft, *aRight = SrcBoxInfo->m_aRRight;
    short *aTop = SrcBoxInfo->m_aRUpper, *aBottom = SrcBoxInfo->m_aRBottom;

    int iThreshold, iCount, iSmall_Idx, iLarge_Idx;
    double dThreRatio = 0.7;

    for (int i = 0; i < SrcBoxInfo->m_iBoundBoxNum; i++) {

        if (SrcBoxInfo->m_ValidBox[i] == false) {
            aValidBoxFlag[i] = false;
            continue;
        }

        size1 = (aRight[i] - aLeft[i]) * (aBottom[i] - aTop[i]);

        for (int j = i; j < SrcBoxInfo->m_iBoundBoxNum; j++) {
            if ((i == j) || (SrcBoxInfo->m_ValidBox[j] == false)) continue;

            //Setting threshold for checking overlapped region size
            size2 = (aRight[j] - aLeft[j]) * (aBottom[j] - aTop[j]);

            if (size1 >= size2) {
                iThreshold = (int)(size2 * dThreRatio);
                iSmall_Idx = j; iLarge_Idx = i;
            }
            else {
                iThreshold = (int)(size1 * dThreRatio);
                iSmall_Idx = i; iLarge_Idx = j;
            }
            //Calculating overlapped region size
            iCount = 0;
            for (int m = aLeft[iSmall_Idx]; m < aRight[iSmall_Idx]; m++) {
                for (int n = aTop[iSmall_Idx]; n < aBottom[iSmall_Idx]; n++) {
                    if (aLeft[iLarge_Idx] <= m && m <= aRight[iLarge_Idx] && aTop[iLarge_Idx] <= n && n <= aBottom[iLarge_Idx]) iCount++;
                }
            }
            //Evaluating overlapped region size
            if (iCount > iThreshold) aValidBoxFlag[iSmall_Idx] = false;
        }
    }

    for (int i = 0; i < SrcBoxInfo->m_iBoundBoxNum; i++) SrcBoxInfo->m_ValidBox[i] = aValidBoxFlag[i];
    delete[] aValidBoxFlag;
}
//-----------------------------------------------------------------------------------------------------------------------------------------//
//														Caculate the overlap region  													   //																							   //
//-----------------------------------------------------------------------------------------------------------------------------------------//
void MultiCues::h_CaclOverlap(BoundingBoxInfo *BoundBoxInfo)
{
    int numberBoxes = BoundBoxInfo->m_iBoundBoxNum;
    for(int i= 0; i < numberBoxes; i++)
    {
        if(BoundBoxInfo->m_ValidBox[i] != false){
            for(int j= i; j < numberBoxes; j++)
            {
                if(i == j || BoundBoxInfo->m_ValidBox[j] == false)
                {
                    continue;
                }
                int xmin = max(BoundBoxInfo->m_aLeft[i],BoundBoxInfo->m_aLeft[j]);
                int xmax = min(BoundBoxInfo->m_aRight[i], BoundBoxInfo->m_aRight[j]);
                int ymin = max(BoundBoxInfo->m_aUpper[i], BoundBoxInfo->m_aUpper[j]);
                int ymax = min(BoundBoxInfo->m_aBottom[i], BoundBoxInfo->m_aBottom[j]);
                int areaJ = (BoundBoxInfo->m_aRight[j] - BoundBoxInfo->m_aLeft[j]) * (BoundBoxInfo->m_aBottom[j] - BoundBoxInfo->m_aUpper[j]);
                int inteArea = max(0, xmax - xmin ) * max(0 , ymax - ymin );
                int areaI = (BoundBoxInfo->m_aRight[i] - BoundBoxInfo->m_aLeft[i]) * (BoundBoxInfo->m_aBottom[i] - BoundBoxInfo->m_aUpper[i]);

                float ratioAreaJ = (float)inteArea / float(areaJ);
                if(ratioAreaJ >= 0.6)
                {
                    BoundBoxInfo->m_ValidBox[j] = false;

                }
                float ratioAreaI =  (float)inteArea / float(areaI);
                if(ratioAreaI >= 0.6)
                {
                    BoundBoxInfo->m_ValidBox[i] = false;
                }
            }
        }
    }
}

//-----------------------------------------------------------------------------------------------------------------------------------------//
//														appearance based verification													   //																							   //
//-----------------------------------------------------------------------------------------------------------------------------------------//
void MultiCues::h_ImAdjust(IplImage *scr, IplImage *dst, Size sz) //, int tol)
{
    bool gama = false;
    int iw = sz.width;
    int ih = sz.height;

    int tol = 1;

    int num0, num1;
    vector<int> hist(256, 0);

    //    dst = scr;
    CvScalar pixel;
    for (int i = 0; i <ih;  i++)
    {
        for(int j = 0; j < iw; j++)
        {
            pixel = cvGet2D(scr, i, j) ; // << endl;
            hist[pixel.val[0]]++;

        }

    }
    //    cv::Mat va= Mat(hist) ; //.size(), 1, CV_32S, hist);
    //    cout << " va " << va << endl;
    vector<int> hist1 = hist;

    for(int i = 1; i < hist.size(); i++)
    {

        //        cout << " hist " << hist[i] << endl;
        hist1[i]= hist1[i-1] + hist[i];

    }
    int total = iw*ih;
    int low_bound = total * tol / 100;
    int high_bound = total * (100-tol) / 100;
    std::vector<int>::iterator low1, low2;
    low1 = lower_bound(hist1.begin(), hist1.end(), low_bound);
    low2 = lower_bound(hist1.begin(), hist1.end(), high_bound);

    num0 = distance(hist1.begin(), low1);
    num1 = distance(hist1.begin(), low2);

    float scale = 255.f / float(num1 - num0);
    CvScalar px;
    Mat temp = cvarrToMat(scr, true);
    //    temp.convertTo(temp, CV_64F);
    Mat1b dt = temp.clone(); //= Mat(cv::Size(iw, ih), CV_8UC1, 0);
    for (int i = 0; i < ih; i++)
    {
        for (int j = 0; j < iw;  j++)
        {
            px = cvGet2D(scr, i, j);
            int vs = max(int(px.val[0]) - num0, 0);
            int vd = min(int(vs*scale + 0.5f), 255);
            //            dst->imageData[j + i * iw]= cv::saturate_cast<uchar>(vd);
            dt(i, j) = cv::saturate_cast<uchar>(vd);
        }
    }

    //    Mat dd = dt;
    IplImage* outim = new IplImage(dt);
    dst = outim;
}

//-----------------------------------------------------------------------------------------------------------------------------------------//
//                                                       Stitching                                                                         //
//-----------------------------------------------------------------------------------------------------------------------------------------//
void MultiCues::h_Stitching(const Mat1b &src, Mat1b &dst, int tol, Vec2i in, Vec2i out)
{
    //}
    //void MultiCues::h_stitching(const Mat1b& src, Mat1b& dst, int tol = 1, Vec2i in = Vec2i(0, 255), Vec2i out = Vec2i(0, 255))
    //{


    dst = src.clone();

    tol = max(0, min(100, tol));
    //    cout << "tol" << tol << endl;

    if (tol > 0)
    {
        // Compute in and out limits

        // Histogram
        vector<int> hist(256, 0);
        for (int r = 0; r < src.rows; ++r) {
            for (int c = 0; c < src.cols; ++c) {
                hist[src(r,c)]++;
            }
        }

        // Cumulative histogram
        vector<int> cum = hist;
        for (int i = 1; i < hist.size(); ++i) {
            cum[i] = cum[i - 1] + hist[i];
        }

        // Compute bounds
        int total = src.rows * src.cols;
        int low_bound = total * tol / 100;
        int upp_bound = total * (100-tol) / 100;
        in[0] = distance(cum.begin(), lower_bound(cum.begin(), cum.end(), low_bound));
        in[1] = distance(cum.begin(), lower_bound(cum.begin(), cum.end(), upp_bound));

    }

    // Stretching
    float scale = float(out[1] - out[0]) / float(in[1] - in[0]);
    for (int r = 0; r < dst.rows; ++r)
    {
        for (int c = 0; c < dst.cols; ++c)
        {
            int vs = max(src(r, c) - in[0], 0);
            int vd = min(int(vs * scale + 0.5f) + out[0], out[1]);
            dst(r, c) = saturate_cast<uchar>(vd);
        }
    }
}
//-----------------------------------------------------------------------------------------------------------------------------------------//
//														appearance based verification													   //																							   //
//-----------------------------------------------------------------------------------------------------------------------------------------//
void MultiCues::EvaluateGhostRegion(IplImage* frame, uchar** aResForeMap, BoundingBoxInfo* BoundBoxInfo) {

    double dThreshold = 7; //10;
    BOOL** aUpdateMap = (BOOL**)malloc(sizeof(BOOL*)*g_iRHeight);
    for (int i = 0; i < g_iRHeight; i++) {
        aUpdateMap[i] = (BOOL*)malloc(sizeof(BOOL)*g_iRWidth);
        for (int j = 0; j < g_iRWidth; j++) aUpdateMap[i][j] = false;
    }
#ifdef DEBUG
    //Step1: Conduct fore-region evaluation to identify ghost regions
    cv::Mat display = cv::cvarrToMat(frame, true);
    string output_dir = "/home/huongnt/Workspace/MovDet/src/multicues/build/ObjectPatch";
    int count = 0;
#endif
    for (int i = 0; i < BoundBoxInfo->m_iBoundBoxNum; i++) {
        if (BoundBoxInfo->m_ValidBox[i] == true) {
            int iWin_w = BoundBoxInfo->m_aRRight[i] - BoundBoxInfo->m_aRLeft[i];
            int iWin_h = BoundBoxInfo->m_aRBottom[i] - BoundBoxInfo->m_aRUpper[i];
            int w = BoundBoxInfo->m_aRight[i] - BoundBoxInfo->m_aLeft[i];
            int h = BoundBoxInfo->m_aBottom[i] - BoundBoxInfo->m_aUpper[i];
            int x = BoundBoxInfo->m_aLeft[i];
            int y = BoundBoxInfo->m_aUpper[i];
#ifdef DEBUG
            cv::Rect  debugrect = cv::Rect(x, y, w , h );
            cv::rectangle(display, debugrect, cv::Scalar(0, 0, 255), 1, 4);
            imshow(" display ",display );
            waitKey(0);
            //Generating edge image from bound-boxed frame region
#endif
            //Generating edge image from aResForeMap
            IplImage* edge_fore = cvCreateImage(cvSize(iWin_w, iWin_h), IPL_DEPTH_8U, 1);
            for (int m = BoundBoxInfo->m_aRUpper[i]; m < BoundBoxInfo->m_aRBottom[i]; m++) {
                for (int n = BoundBoxInfo->m_aRLeft[i]; n < BoundBoxInfo->m_aRRight[i]; n++) {
                    edge_fore->imageData[(m - BoundBoxInfo->m_aRUpper[i])*edge_fore->widthStep + (n - BoundBoxInfo->m_aRLeft[i])] = (char)aResForeMap[m][n];
                }
            }
#ifdef RESIZE
            IplImage* resized_frame = cvCreateImage(cvSize(g_iRWidth, g_iRHeight), IPL_DEPTH_8U, 1);
            cvResize(frame, resized_frame, CV_INTER_NN);

            cvSetImageROI(resized_frame, cvRect(BoundBoxInfo->m_aRLeft[i], BoundBoxInfo->m_aRUpper[i], iWin_w, iWin_h));
            IplImage* edge_frame = cvCreateImage(cvSize(iWin_w, iWin_h), IPL_DEPTH_8U, 1);

//            cvCvtColor(resized_frame, edge_frame, CV_BGR2GRAY);
            cvCopy(resized_frame, edge_frame);
            cvResetImageROI(resized_frame);
            cvReleaseImage(&resized_frame);

#else
            cvSetImageROI(frame, cvRect(BoundBoxInfo->m_aLeft[i], BoundBoxInfo->m_aUpper[i], w, h));
            IplImage* edge_frame = cvCreateImage(cvSize(w, h), IPL_DEPTH_8U, 1);

            cvCvtColor(frame, edge_frame, CV_BGR2GRAY);
            cvResetImageROI(frame);

            // Edge from multicue map
            IplImage* edge_fore_sz = cvCreateImage(cvSize(w, h), IPL_DEPTH_8U, 1);
            cvResize(edge_fore, edge_fore_sz);
            edge_fore = edge_fore_sz;
#endif
            CvScalar mean, stdev;
            cvAvgSdv(edge_frame, &mean, &stdev, NULL);
            int ih = edge_frame->height;
            int iw = edge_frame->width;

            IplImage* thresEdge = cvCreateImage(cvSize(iw, ih), IPL_DEPTH_8U, 1) ;
            cvThreshold(edge_frame, thresEdge, mean.val[0] + stdev.val[0] , 255, CV_THRESH_TOZERO);

            IplConvKernel *convKernel = cvCreateStructuringElementEx(3, 3, 1, 1, CV_SHAPE_ELLIPSE);
            cvMorphologyEx(thresEdge, thresEdge, NULL, convKernel,  CV_MOP_OPEN, 1 );
            cvMorphologyEx(thresEdge,thresEdge, NULL, convKernel, CV_MOP_CLOSE, 1);

            cvCanny(thresEdge, thresEdge, 50, 100);
            cvCanny(edge_fore, edge_fore, 50, 100);

#ifdef DEBUG
            Mat thresMedianMat = cvarrToMat(thresEdge, true);
            imshow("thresmediaMat", thresMedianMat);

            cv::Mat thresMat = cvarrToMat(edge_frame, true);
            cv::imshow("stitMat", thresMat);
            cv::waitKey(0);
            cv::Mat display_thres = cvarrToMat(thresEdge, true);
            cv::Mat display_fore = cvarrToMat(edge_fore, true);

            imshow(" display_thres ",display_thres);
            imshow(" display_fore ", display_fore);
            waitKey(0);
#endif
            // Calculate HausdorDist
            double distance = CalculateHausdorffDist(thresEdge, edge_fore);
            //            cout << "Distance" << distance << endl;
            if (distance > dThreshold ){
                for (int m = BoundBoxInfo->m_aRUpper[i]; m < BoundBoxInfo->m_aRBottom[i]; m++) {
                    for (int n = BoundBoxInfo->m_aRLeft[i]; n < BoundBoxInfo->m_aRRight[i]; n++) {
                        aUpdateMap[m][n] = true;
                    }
                }
            }
            cvReleaseImage(&edge_frame);
            cvReleaseImage(&edge_fore);
        }
    }

#ifdef GAUSS
    h_UpdateGaussBG(frame, h_BGModel, BoundBoxInfo, h_GrayFrame, aUpdateMap, aResForeMap);
#endif
    //Step2: Adding information fo ghost region pixels to background model
    float fLearningRate = g_fLearningRate;

    for (int i = 0; i < g_iRHeight; i++) {
        for (int j = 0; j < g_iRWidth; j++) {

            if (aUpdateMap[i][j] == true) {
                point center;
                center.m_nX = j; center.m_nY = i;

                T_ModelConstruction(g_nTextureTrainVolRange, fLearningRate, h_GrayFrame, center, g_aNeighborDirection[i][j], g_TextureModel[i][j]);
                C_CodebookConstruction(h_GrayFrame, j, i, g_nColorTrainVolRange, fLearningRate, g_ColorModel[i][j]);

                T_ClearNonEssentialEntries(g_iBackClearPeriod, g_TextureModel[i][j]);
                C_ClearNonEssentialEntries(g_iBackClearPeriod, g_ColorModel[i][j]);
            }
        }
    }

    for (int i = 0; i < g_iRHeight; i++) free(aUpdateMap[i]);
    free(aUpdateMap);
}

//-----------------------------------------------------------------------------------------------------------------------------------------//
//								the function to calculate partial undirected Hausdorff distance(forward distance)						   //																							   //
//-----------------------------------------------------------------------------------------------------------------------------------------//
double MultiCues::CalculateHausdorffDist(IplImage* input_image, IplImage* model_image) {

    //Step1: Generating imag vectors
    //For reduce errors, points at the image boundary are excluded
    std::vector<point> vInput, vModel;
    point temp;

    //input image --> input vector
    for (int i = 0; i < input_image->height; i++) {
        for (int j = 0; j < input_image->width; j++) {

            if ((uchar)input_image->imageData[i*input_image->widthStep + j] == 0) continue;

            temp.m_nX = j; temp.m_nY = i;
            vInput.push_back(temp);
        }
    }
    //model image --> model vector
    for (int i = 0; i < model_image->height; i++) {
        for (int j = 0; j < model_image->width; j++) {
            if ((uchar)model_image->imageData[i*model_image->widthStep + j] == 0) continue;

            temp.m_nX = j; temp.m_nY = i;
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
            temp1 = (*iter_m).m_nX - (*iter_i).m_nX;
            temp2 = (*iter_m).m_nY - (*iter_i).m_nY;
            dDist = temp1*temp1 + temp2*temp2;

            if (dDist < dMinDist) dMinDist = dDist;
        }
        vTempDist.push_back(dMinDist);
    }
    sort(vTempDist.begin(), vTempDist.end()); //in ascending order

    double dQuantileVal = 0.9, dForwardDistance;
    int iDistIndex = (int)(dQuantileVal*vTempDist.size());
    if (iDistIndex == vTempDist.size()) iDistIndex -= 1;

    dForwardDistance = sqrt(vTempDist[iDistIndex]);
    return dForwardDistance;
}


//-----------------------------------------------------------------------------------------------------------------------------------------//
//											function to remove non-valid bounding boxes fore fore-candidates							   //
//-----------------------------------------------------------------------------------------------------------------------------------------//
void MultiCues::RemovingInvalidForeRegions(uchar** aResForeMap, BoundingBoxInfo* BoundBoxInfo) {
#ifdef DEBUG
    Mat aResForeMat1 = cv::Mat(cv::Size(g_iRWidth, g_iRHeight), CV_8UC1, 1);
    Mat background1 = cv::Mat(cv::Size(g_iRWidth, g_iRHeight), CV_8UC1, 1);
    for(int i = 0 ; i < g_iRHeight; i++)
    {
        for(int j = 0; j < g_iRWidth; j++)
        {
            aResForeMat1.at<uchar>(i, j) = (uchar) aResForeMap[i][j];
        }
    }

    imshow("fgMapbefore", aResForeMat1);
    waitKey(0);
#endif
    int iBoxNum = BoundBoxInfo->m_iBoundBoxNum;

    for (int k = 0; k < iBoxNum; k++) {
        if (BoundBoxInfo->m_ValidBox[k] == false) {
            for (int i = BoundBoxInfo->m_aRUpper[k]; i < BoundBoxInfo->m_aRBottom[k]; i++) {
                for (int j = BoundBoxInfo->m_aRLeft[k]; j < BoundBoxInfo->m_aRRight[k]; j++) {
                    if (aResForeMap[i][j] == 255)
                    {
                        aResForeMap[i][j] =(uchar)0;
                    }

                }
            }
        }

    }
#define OPTMASK
#ifdef OPTMASK


#endif
#ifdef DEBUG
    Mat aResForeMat = cv::Mat(cv::Size(g_iRWidth, g_iRHeight), CV_8UC1, 1);
    for(int i = 0 ; i < g_iRHeight; i++)
    {
        for(int j = 0; j < g_iRWidth; j++)
        {
            aResForeMat.at<uchar>(i, j) = (uchar) aResForeMap[i][j];
        }
    }
    string afterGauss = DEBUGPATH + "/afGauss.jpg";
    imwrite(afterGauss, aResForeMat);
    imshow("fgMapAfter", aResForeMat);
    waitKey(0);
#endif
}

//-----------------------------------------------------------------------------------------------------------------------------------------//
//													the function returning a foreground binary-map										   //
//-----------------------------------------------------------------------------------------------------------------------------------------//
void MultiCues::GetForegroundMap(IplImage* return_image, IplImage* input_frame) {

    if (g_bForegroundMapEnable == false) return;

    IplImage* temp_image = cvCreateImage(cvSize(g_iRWidth, g_iRHeight), IPL_DEPTH_8U, 1);
    //    Mat  check_input = cv::cvarrToMat(input_frame, true);
    //    cv::imshow("inputframe", check_input);
    //    cv::waitKey(0);
    //	if (input_frame == NULL) {
    for (int i = 0; i < g_iRHeight; i++) {
        for (int j = 0; j < g_iRWidth; j++) {
            temp_image->imageData[i*temp_image->widthStep + j] = (uchar)g_aResizedForeMap[i][j];
        }
    }
    //        for (int i = 0; i < g_iRHeight; i++) {
    //            for (int j = 0; j < g_iRWidth; j++) {
    //                if((int)temp_image->imageData[i * temp_image->widthStep + j] != 0)
    //                {
    //                  bg_image[i][j] = 0;
    //                }
    //                bg_image[i][j] = (uchar)input_frame->imageData[i + temp_image->widthStep +j];

    //            }
    //        }
    //	}
    //	else {

    //		cvResize(input_frame, temp_image);
    //        int val = 255;
    //		for (int i = 0; i < g_iRHeight; i++) {
    //			for (int j = 0; j < g_iRWidth; j++) {

    //				if (g_aResizedForeMap[i][j] == 255) {
    //                    uchar B = (uchar)temp_image->imageData[i*temp_image->widthStep + j];
    //                    B = (uchar)(B*0.45 + val * 0.55);
    //                    temp_image->imageData[i*temp_image->widthStep + j] = (char)B;
    ////                    bg_image[i][j] = 255;
    //                }
    ////                bg_image[i][j] = (uchar)input_frame->imageData[i*temp_image->widthStep + j];
    //			}
    //		}
    //	}

    cvResize(temp_image, return_image);
    cvReleaseImage(&temp_image);
}

//-----------------------------------------------------------------------------------------------------------------------------------------//
//												the	initialization function for the texture-models									       //
//-----------------------------------------------------------------------------------------------------------------------------------------//
void MultiCues::T_AllocateTextureModelRelatedMemory() {
    int i, j, k;

    //neighborhood system related
    int iMaxNeighborArraySize = 8;
    g_aNeighborDirection = (point***)malloc(sizeof(point**)*g_iRHeight);
    for (i = 0; i < g_iRHeight; i++) {
        g_aNeighborDirection[i] = (point**)malloc(sizeof(point*)*g_iRWidth);
        for (j = 0; j < g_iRWidth; j++) {
            g_aNeighborDirection[i][j] = (point*)malloc(sizeof(point)*iMaxNeighborArraySize);
        }
    }
    T_SetNeighborDirection(g_aNeighborDirection);
//    for(int y = 0; y < g_iRHeight; y ++)
//    {
//        for(int x = 0; x < g_iRWidth; x++)
//        {
//            for(int k = 0; k < g_nNeighborNum; k++)
//            {
//                printf("(%d, %d)\t", g_aNeighborDirection[y][x][k].m_nX,g_aNeighborDirection[y][x][k].m_nY );
//            }
//        }
//        printf("\n");
//    }
    //texture-model related
    int iElementArraySize = 6;
    g_TextureModel = (TextureModel****)malloc(sizeof(TextureModel***)*g_iRHeight);
    for (i = 0; i < g_iRHeight; i++) {
        g_TextureModel[i] = (TextureModel***)malloc(sizeof(TextureModel**)*g_iRWidth);
        for (j = 0; j < g_iRWidth; j++) {
            g_TextureModel[i][j] = (TextureModel**)malloc(sizeof(TextureModel*)*g_nNeighborNum);
            for (k = 0; k < g_nNeighborNum; k++) {
                g_TextureModel[i][j][k] = (TextureModel*)malloc(sizeof(TextureModel));
                g_TextureModel[i][j][k]->m_Codewords = (TextureCodeword**)malloc(sizeof(TextureCodeword*)*iElementArraySize);
                g_TextureModel[i][j][k]->m_iElementArraySize = iElementArraySize;
                g_TextureModel[i][j][k]->m_iNumEntries = 0;
                g_TextureModel[i][j][k]->m_iTotal = 0;
                g_TextureModel[i][j][k]->m_bID = 1;
            }
        }
    }
    //    int iElementArraySize = 6;
    h_TextureModel = (TextureModel****)malloc(sizeof(TextureModel***)*g_iRHeight);
    for (i = 0; i < g_iRHeight; i++) {
        h_TextureModel[i] = (TextureModel***)malloc(sizeof(TextureModel**)*g_iRWidth);
        for (j = 0; j < g_iRWidth; j++) {
            h_TextureModel[i][j] = (TextureModel**)malloc(sizeof(TextureModel*)*g_nNeighborNum);
            for (k = 0; k < g_nNeighborNum; k++) {
                h_TextureModel[i][j][k] = (TextureModel*)malloc(sizeof(TextureModel));
                h_TextureModel[i][j][k]->m_Codewords = (TextureCodeword**)malloc(sizeof(TextureCodeword*)*iElementArraySize);
                h_TextureModel[i][j][k]->m_iElementArraySize = iElementArraySize;
                h_TextureModel[i][j][k]->m_iNumEntries = 0;
                h_TextureModel[i][j][k]->m_iTotal = 0;
                h_TextureModel[i][j][k]->m_bID = 1;
            }
        }
    }
    // inint Gmodel

    g_aTextureConfMap = (float**)malloc(sizeof(float*)*g_iRHeight);
    for (i = 0; i < g_iRHeight; i++) g_aTextureConfMap[i] = (float*)malloc(sizeof(float)*g_iRWidth);

    //cache-book related
    if (g_bAbsorptionEnable == true) {
        iElementArraySize = iElementArraySize / 2;
        if (iElementArraySize < 3)iElementArraySize = 3;

        g_TCacheBook = (TextureModel****)malloc(sizeof(TextureModel***)*g_iRHeight);
        for (i = 0; i < g_iRHeight; i++) {
            g_TCacheBook[i] = (TextureModel***)malloc(sizeof(TextureModel**)*g_iRWidth);
            for (j = 0; j < g_iRWidth; j++) {
                g_TCacheBook[i][j] = (TextureModel**)malloc(sizeof(TextureModel*)*g_nNeighborNum);
                for (k = 0; k < g_nNeighborNum; k++) {
                    g_TCacheBook[i][j][k] = (TextureModel*)malloc(sizeof(TextureModel));
                    g_TCacheBook[i][j][k]->m_Codewords = (TextureCodeword**)malloc(sizeof(TextureCodeword*)*iElementArraySize);
                    g_TCacheBook[i][j][k]->m_iElementArraySize = iElementArraySize;
                    g_TCacheBook[i][j][k]->m_iNumEntries = 0;
                    g_TCacheBook[i][j][k]->m_iTotal = 0;
                    g_TCacheBook[i][j][k]->m_bID = 0;
                }
            }
        }

        g_aTReferredIndex = (short***)malloc(sizeof(short**)*g_iRHeight);
        g_aTContinuousCnt = (short***)malloc(sizeof(short**)*g_iRHeight);
        for (i = 0; i < g_iRHeight; i++) {
            g_aTReferredIndex[i] = (short**)malloc(sizeof(short*)*g_iRWidth);
            g_aTContinuousCnt[i] = (short**)malloc(sizeof(short*)*g_iRWidth);
            for (j = 0; j < g_iRWidth; j++) {
                g_aTReferredIndex[i][j] = (short*)malloc(sizeof(short)*g_nNeighborNum);
                g_aTContinuousCnt[i][j] = (short*)malloc(sizeof(short)*g_nNeighborNum);
                for (k = 0; k < g_nNeighborNum; k++) {
                    g_aTReferredIndex[i][j][k] = -1;
                    g_aTContinuousCnt[i][j][k] = 0;
                }
            }
        }
    }
}
//-----------------------------------------------------------------------------------------------------------------------------------------//
//															the memory release function											           //
//-----------------------------------------------------------------------------------------------------------------------------------------//
void MultiCues::T_ReleaseTextureModelRelatedMemory() {
    int i, j, k, m;
    short nNeighborNum = g_nNeighborNum;

    for (i = 0; i < g_iRHeight; i++) {
        for (j = 0; j < g_iRWidth; j++) {
            for (k = 0; k < nNeighborNum; k++) {
                for (m = 0; m < g_TextureModel[i][j][k]->m_iNumEntries; m++) free(g_TextureModel[i][j][k]->m_Codewords[m]);
                free(g_TextureModel[i][j][k]->m_Codewords);
                free(g_TextureModel[i][j][k]);
            }
            free(g_TextureModel[i][j]);
        }free(g_TextureModel[i]);
    }
    free(g_TextureModel);

    for (i = 0; i < g_iRHeight; i++) {
        for (j = 0; j < g_iRWidth; j++) {
            for (k = 0; k < nNeighborNum; k++) {
                for (m = 0; m < h_TextureModel[i][j][k]->m_iNumEntries; m++) free(h_TextureModel[i][j][k]->m_Codewords[m]);
                free(h_TextureModel[i][j][k]->m_Codewords);
                free(h_TextureModel[i][j][k]);
            }
            free(h_TextureModel[i][j]);
        }free(h_TextureModel[i]);
    }
    free(h_TextureModel);
    for (i = 0; i < g_iRHeight; i++) {
        for (j = 0; j < g_iRWidth; j++) {

            free(h_BGModel[i][j]);
        }

    }

    for (i = 0; i < g_iRHeight; i++) {
        for (j = 0; j < g_iRWidth; j++) free(g_aNeighborDirection[i][j]);
        free(g_aNeighborDirection[i]);
    }
    free(g_aNeighborDirection);

    for (i = 0; i < g_iRHeight; i++) free(g_aTextureConfMap[i]);
    free(g_aTextureConfMap);

    if (g_bAbsorptionEnable == true) {
        for (i = 0; i < g_iRHeight; i++) {
            for (j = 0; j < g_iRWidth; j++) {
                for (k = 0; k < nNeighborNum; k++) {
                    for (m = 0; m < g_TCacheBook[i][j][k]->m_iNumEntries; m++) free(g_TCacheBook[i][j][k]->m_Codewords[m]);
                    free(g_TCacheBook[i][j][k]->m_Codewords);
                    free(g_TCacheBook[i][j][k]);
                }
                free(g_TCacheBook[i][j]);
            }free(g_TCacheBook[i]);
        }
        free(g_TCacheBook);

        for (i = 0; i < g_iRHeight; i++) {
            for (j = 0; j < g_iRWidth; j++) {
                free(g_aTReferredIndex[i][j]);
                free(g_aTContinuousCnt[i][j]);
            }
            free(g_aTReferredIndex[i]);
            free(g_aTContinuousCnt[i]);
        }
        free(g_aTReferredIndex);
        free(g_aTContinuousCnt);

    }
}


//
//-----------------------------------------------------------------------------------------------------------------------------------------//
//												Huong :  T_ModelConstruction                        					                   //																									   //
//-----------------------------------------------------------------------------------------------------------------------------------------//


void MultiCues::T_ModelConstruction(short nTrainVolRange, float fLearningRate,
                                    uchar** gray, point center,
                                    point* aNei, TextureModel** aModel) {
    int i, j;
    int iMatchedIndex;

    short nNeighborNum = g_nNeighborNum;

    float fDifference;
    float fDiffMean;

    float fNegLearningRate = 1 - fLearningRate;

    //for all neighboring pairs
    for (i = 0; i < nNeighborNum; i++) {

        //		fDifference = (float)(aXYZ[center.m_nY][center.m_nX][2] - aXYZ[aNei[i].m_nY][aNei[i].m_nX][2]);
        fDifference = (float)(gray[center.m_nY][center.m_nX]- gray[aNei[i].m_nY][aNei[i].m_nX]);

        //Step1: matching
        iMatchedIndex = -1;
        for (j = 0; j < aModel[i]->m_iNumEntries; j++) {
            if (aModel[i]->m_Codewords[j]->m_fLowThre <= fDifference && fDifference <= aModel[i]->m_Codewords[j]->m_fHighThre) {
                iMatchedIndex = j;
                break;
            }
        }

        aModel[i]->m_iTotal++;
        //Step2: adding a new element
        if (iMatchedIndex == -1) {
            //element array
            if (aModel[i]->m_iElementArraySize == aModel[i]->m_iNumEntries) {
                aModel[i]->m_iElementArraySize += 5;
                TextureCodeword **temp = (TextureCodeword**)malloc(sizeof(TextureCodeword*)*aModel[i]->m_iElementArraySize);
                for (j = 0; j < aModel[i]->m_iNumEntries; j++) {
                    temp[j] = aModel[i]->m_Codewords[j];
                    aModel[i]->m_Codewords[j] = NULL;
                }
                free(aModel[i]->m_Codewords);
                aModel[i]->m_Codewords = temp;
            }

            aModel[i]->m_Codewords[aModel[i]->m_iNumEntries] = (TextureCodeword*)malloc(sizeof(TextureCodeword));
            aModel[i]->m_Codewords[aModel[i]->m_iNumEntries]->m_fMean = fDifference;
            aModel[i]->m_Codewords[aModel[i]->m_iNumEntries]->m_fLowThre = aModel[i]->m_Codewords[aModel[i]->m_iNumEntries]->m_fMean - nTrainVolRange;
            aModel[i]->m_Codewords[aModel[i]->m_iNumEntries]->m_fHighThre = aModel[i]->m_Codewords[aModel[i]->m_iNumEntries]->m_fMean + nTrainVolRange;

            aModel[i]->m_Codewords[aModel[i]->m_iNumEntries]->m_iT_first_time = aModel[i]->m_iTotal;
            aModel[i]->m_Codewords[aModel[i]->m_iNumEntries]->m_iT_last_time = aModel[i]->m_iTotal;
            aModel[i]->m_Codewords[aModel[i]->m_iNumEntries]->m_iMNRL = aModel[i]->m_iTotal - 1;
            aModel[i]->m_iNumEntries++;
            //            cout << "check number entries after: " << aModel[i]->m_iNumEntries << endl;
        }
        //Step3: update
        else {

            fDiffMean = aModel[i]->m_Codewords[iMatchedIndex]->m_fMean;
            aModel[i]->m_Codewords[iMatchedIndex]->m_fMean = fLearningRate*fDifference + fNegLearningRate *fDiffMean;
            aModel[i]->m_Codewords[iMatchedIndex]->m_fLowThre = aModel[i]->m_Codewords[iMatchedIndex]->m_fMean - nTrainVolRange;
            aModel[i]->m_Codewords[iMatchedIndex]->m_fHighThre = aModel[i]->m_Codewords[iMatchedIndex]->m_fMean + nTrainVolRange;
            aModel[i]->m_Codewords[iMatchedIndex]->m_iT_last_time = aModel[i]->m_iTotal;
        }

        //cache-book handling
        // Background handing
        if (aModel[i]->m_bID == 1) {
            //1. m_iMNRL update
            int negTime;
            for (j = 0; j < aModel[i]->m_iNumEntries; j++) {
                //m_iMNRL update
                negTime = aModel[i]->m_iTotal - aModel[i]->m_Codewords[j]->m_iT_last_time + aModel[i]->m_Codewords[j]->m_iT_first_time - 1;
                if (aModel[i]->m_Codewords[j]->m_iMNRL < negTime) aModel[i]->m_Codewords[j]->m_iMNRL = negTime;
            }

            //2. g_aTReferredIndex[center.m_nY][center.m_nX][i] update
            if (g_bAbsorptionEnable == true) g_aTReferredIndex[center.m_nY][center.m_nX][i] = -1;
        }
        // Foreground handing

        else {
            //1. m_iMNRL update
            if (iMatchedIndex == -1) aModel[i]->m_Codewords[aModel[i]->m_iNumEntries - 1]->m_iMNRL = 0;

            //2. g_aTReferredIndex[center.m_nY][center.m_nX][i] update
            if (iMatchedIndex == -1) {
                g_aTReferredIndex[center.m_nY][center.m_nX][i] = aModel[i]->m_iNumEntries - 1;
                g_aTContinuousCnt[center.m_nY][center.m_nX][i] = 1;
            }
            else {
                if (iMatchedIndex == g_aTReferredIndex[center.m_nY][center.m_nX][i]) g_aTContinuousCnt[center.m_nY][center.m_nX][i]++;
                else {
                    g_aTReferredIndex[center.m_nY][center.m_nX][i] = iMatchedIndex;
                    g_aTContinuousCnt[center.m_nY][center.m_nX][i] = 1;
                }
            }
        }
    }

}

//-----------------------------------------------------------------------------------------------------------------------------------------//
//												Clear non-essential codewords of the given codebook						                   //																									   //
//-----------------------------------------------------------------------------------------------------------------------------------------//
void MultiCues::T_ClearNonEssentialEntries(short nClearNum, TextureModel** aModel) {
    int i, n;
    int iStaleThresh = (int)(nClearNum*0.5);
    int iKeepCnt;
    int* aKeep;

    short nNeighborNum = g_nNeighborNum;

    TextureModel* c;

    for (n = 0; n < nNeighborNum; n++) {
        c = aModel[n];

        if (c->m_iTotal < nClearNum) continue; //(being operated only when c[w][h]->m_iTotal == nClearNum)

        //Step1: initialization
        aKeep = (int*)malloc(sizeof(int)*c->m_iNumEntries);

        iKeepCnt = 0;


        //Step2: Find non-essential code-words
        for (i = 0; i < c->m_iNumEntries; i++) {
            if (c->m_Codewords[i]->m_iMNRL > iStaleThresh) {
                aKeep[i] = 0; //removal candidate
            }
            else {
                aKeep[i] = 1;
                iKeepCnt++;
            }
        }

        //Step3: Perform removal
        if (iKeepCnt == 0 || iKeepCnt == c->m_iNumEntries) {
            for (i = 0; i < c->m_iNumEntries; i++) {
                c->m_Codewords[i]->m_iT_first_time = 1;
                c->m_Codewords[i]->m_iT_last_time = 1;
                c->m_Codewords[i]->m_iMNRL = 0;
            }
        }

        else {
            iKeepCnt = 0;
            TextureCodeword** temp = (TextureCodeword**)malloc(sizeof(TextureCodeword*)*c->m_iNumEntries);

            for (i = 0; i < c->m_iNumEntries; i++) {
                if (aKeep[i] == 1) {
                    temp[iKeepCnt] = c->m_Codewords[i];
                    temp[iKeepCnt]->m_iT_first_time = 1;
                    temp[iKeepCnt]->m_iT_last_time = 1;
                    temp[iKeepCnt]->m_iMNRL = 0;
                    iKeepCnt++;
                }
                else free(c->m_Codewords[i]);
            }

            //ending..
            free(c->m_Codewords);
            c->m_Codewords = temp;
            c->m_iElementArraySize = c->m_iNumEntries;
            c->m_iNumEntries = iKeepCnt;
        }
        c->m_iTotal = 0;
        free(aKeep);

    }

}

//-----------------------------------------------------------------------------------------------------------------------------------------//
//								Clear non-essential codewords of the given codebook (only for the cache-book)			                   //
//-----------------------------------------------------------------------------------------------------------------------------------------//
void MultiCues::T_ClearNonEssentialEntriesForCachebook(uchar bLandmark, short* nReferredIdxArr, short nClearNum, TextureModel** pCachebook) {
    int i, n;
    short nNeighborNum = g_nNeighborNum;

    TextureModel* c;
    short nReferredIdx;

    for (n = 0; n < nNeighborNum; n++) {

        c = pCachebook[n];
        nReferredIdx = nReferredIdxArr[n];

        //pCachebook->m_iTotal < nClearNum? --> MNRL update
        if (c->m_iTotal < nClearNum) {
            for (i = 0; i < c->m_iNumEntries; i++) {
                if (bLandmark == 255 && i == nReferredIdx) c->m_Codewords[i]->m_iMNRL = 0;
                else c->m_Codewords[i]->m_iMNRL++;
            }

            c->m_iTotal++;
        }

        //Perform clearing
        else {
            int iStaleThreshold = 5;

            int* aKeep;
            short nKeepCnt;

            aKeep = (int*)malloc(sizeof(int)*c->m_iNumEntries);
            nKeepCnt = 0;

            for (i = 0; i < c->m_iNumEntries; i++) {
                if (c->m_Codewords[i]->m_iMNRL < iStaleThreshold) {
                    aKeep[i] = 1;
                    nKeepCnt++;
                }
                else aKeep[i] = 0;
            }

            c->m_iElementArraySize = nKeepCnt + 2;
            if (c->m_iElementArraySize < 3) c->m_iElementArraySize = 3;

            TextureCodeword** temp = (TextureCodeword**)malloc(sizeof(TextureCodeword*)*c->m_iElementArraySize);
            nKeepCnt = 0;

            for (i = 0; i < c->m_iNumEntries; i++) {
                if (aKeep[i] == 1) {
                    temp[nKeepCnt] = c->m_Codewords[i];
                    temp[nKeepCnt]->m_iMNRL = 0;
                    nKeepCnt++;
                }
                else {
                    free(c->m_Codewords[i]);
                }

            }

            //ending..
            free(c->m_Codewords);
            c->m_Codewords = temp;
            c->m_iNumEntries = nKeepCnt;
            c->m_iTotal = 0;

            free(aKeep);
        }
    }

}

//-----------------------------------------------------------------------------------------------------------------------------------------//
//											Huongnt382 thay doi                             				                               //
//-----------------------------------------------------------------------------------------------------------------------------------------//
void MultiCues::T_GetConfidenceMap_Par(uchar** gray, float** aTextureMap, point*** aNeiDirArr, TextureModel**** aModel) {

    int iBound_w = g_iRWidth - g_nRadius;
    int iBound_h = g_iRHeight - g_nRadius;

    short nNeighborNum = g_nNeighborNum;
    float fPadding = 5;

    for (int h = 0; h < g_iRHeight; h++) {
        for (int w = 0; w < g_iRWidth; w++) {

            if (h < g_nRadius || h >= iBound_h || w < g_nRadius || w >= iBound_w) {
                aTextureMap[h][w] = 0;
                continue;
            }

            int nMatchedCount = 0;
            float fDiffSum = 0;
            float fDifference;
            point nei;

            for (int i = 0; i < nNeighborNum; i++) {

                nei.m_nX = aNeiDirArr[h][w][i].m_nX;
                nei.m_nY = aNeiDirArr[h][w][i].m_nY;

                fDifference = (float)(gray[h][w] - gray[nei.m_nY][nei.m_nX]);
                if (fDifference < 0) fDiffSum -= fDifference;
                else fDiffSum += fDifference;

                for (int j = 0; j < aModel[h][w][i]->m_iNumEntries; j++) {
                    if (aModel[h][w][i]->m_Codewords[j]->m_fLowThre - fPadding <= fDifference && fDifference <= aModel[h][w][i]->m_Codewords[j]->m_fHighThre + fPadding) {
                        nMatchedCount++;
                        break;
                    }
                }
            }
            aTextureMap[h][w] = 1 - (float)nMatchedCount / nNeighborNum;
        }
        //            else aTextureMap[h][w] = 0;
        //#ifdef CMD
        //        }
        //#endif

    }
#ifdef DEBUG
    IplImage* display;
    display =  cvCreateImage(cvSize(g_iRWidth, g_iRHeight),IPL_DEPTH_8U, 1);
    for (int h = 0; h < g_iRHeight; h++)
    {
        for(int w = 0; w < g_iRWidth; w++)
        {
            if (aTextureMap[h][w] != 0)
            {
                display->imageData[w + h* g_iRWidth] = 255;
            }
        }

    }
    cv::Mat displayMat = cv::cvarrToMat(display, true);
    string texturersepose = DEBUGPATH + "/texture_Respone.jpg";
    cv::imwrite(texturersepose, displayMat);
    cv::imshow(" Texture response ", displayMat);
    cv::waitKey(0);
#endif
}
//-----------------------------------------------------------------------------------------------------------------------------------------//
//											Absorbing Ghost Non-background Region Update					                               //
//-----------------------------------------------------------------------------------------------------------------------------------------//
void MultiCues::T_Absorption(int iAbsorbCnt, point pos, short*** aContinuCnt, short*** aRefferedIndex, TextureModel** pModel, TextureModel** pCache) {
    int i, j, k;
    int iLeavingIndex;

    //short g_nRadius = 2;
    short nNeighborNum = g_nNeighborNum;

    for (i = 0; i < nNeighborNum; i++) {
        //set iLeavingIndex
        if (aContinuCnt[pos.m_nY][pos.m_nX][i] < iAbsorbCnt) continue;

        iLeavingIndex = aRefferedIndex[pos.m_nY][pos.m_nX][i];

        //array expansion
        if (pModel[i]->m_iElementArraySize == pModel[i]->m_iNumEntries) {
            pModel[i]->m_iElementArraySize = pModel[i]->m_iElementArraySize + 5;
            TextureCodeword** temp = (TextureCodeword**)malloc(sizeof(TextureCodeword*)*pModel[i]->m_iElementArraySize);
            for (j = 0; j < pModel[i]->m_iNumEntries; j++) temp[j] = pModel[i]->m_Codewords[j];
            free(pModel[i]->m_Codewords);
            pModel[i]->m_Codewords = temp;
        }

        //movement from the cache-book to the codebook
        pModel[i]->m_Codewords[pModel[i]->m_iNumEntries] = pCache[i]->m_Codewords[iLeavingIndex];

        pModel[i]->m_iTotal = pModel[i]->m_iTotal + 1;

        pModel[i]->m_Codewords[pModel[i]->m_iNumEntries]->m_iT_first_time = pModel[i]->m_iTotal;
        pModel[i]->m_Codewords[pModel[i]->m_iNumEntries]->m_iT_last_time = pModel[i]->m_iTotal;
        pModel[i]->m_Codewords[pModel[i]->m_iNumEntries]->m_iMNRL = pModel[i]->m_iTotal - 1;
        pModel[i]->m_iNumEntries = pModel[i]->m_iNumEntries + 1;

        k = 0;
        TextureCodeword **temp_Cache = (TextureCodeword**)malloc(sizeof(TextureCodeword*)*pCache[i]->m_iElementArraySize);
        for (j = 0; j < pCache[i]->m_iNumEntries; j++) {
            if (j == iLeavingIndex) continue;
            else {
                temp_Cache[k] = pCache[i]->m_Codewords[j];
                k++;
            }
        }
        free(pCache[i]->m_Codewords);
        pCache[i]->m_Codewords = temp_Cache;
        pCache[i]->m_iNumEntries = k;
    }
}

//-----------------------------------------------------------------------------------------------------------------------------------------//
//													the function to set neighborhood system												   //
//-----------------------------------------------------------------------------------------------------------------------------------------//
void MultiCues::T_SetNeighborDirection(point*** aNeighborPos) {
    int i, j, k;
    point* aSearchDirection = (point*)malloc(sizeof(point)*g_nNeighborNum);

    aSearchDirection[0].m_nX = -2;//180 degree
    aSearchDirection[0].m_nY = 0;

    aSearchDirection[1].m_nX = -1;//123 degree
    aSearchDirection[1].m_nY = -2;

    aSearchDirection[2].m_nX = 1;//45 degree
    aSearchDirection[2].m_nY = -2;

    aSearchDirection[3].m_nX = 2;//0 degree
    aSearchDirection[3].m_nY = 0;

    aSearchDirection[4].m_nX = 1;//-45 degree
    aSearchDirection[4].m_nY = 2;

    aSearchDirection[5].m_nX = -1;//-135 degree
    aSearchDirection[5].m_nY = 2;

    point temp_pos;

    for (i = 0; i < g_iRHeight; i++) {
        for (j = 0; j < g_iRWidth; j++) {
            for (k = 0; k < g_nNeighborNum; k++) {
                temp_pos.m_nX = j + aSearchDirection[k].m_nX;
                temp_pos.m_nY = i + aSearchDirection[k].m_nY;

                if (temp_pos.m_nX < 0 || temp_pos.m_nX >= g_iRWidth || temp_pos.m_nY < 0 || temp_pos.m_nY >= g_iRHeight) {
                    aNeighborPos[i][j][k].m_nX = -1;
                    aNeighborPos[i][j][k].m_nY = -1;
                }

                else {
                    aNeighborPos[i][j][k].m_nX = temp_pos.m_nX;
                    aNeighborPos[i][j][k].m_nY = temp_pos.m_nY;
                }
            }
        }
    }
    free(aSearchDirection);
}

//-----------------------------------------------------------------------------------------------------------------------------------------//
//													the color-model initialization function												   //
//-----------------------------------------------------------------------------------------------------------------------------------------//
void MultiCues::C_AllocateColorModelRelatedMemory() {
    int i, j;

    int iElementArraySize = 10;

    //codebook initialization
    g_ColorModel = (ColorModel***)malloc(sizeof(ColorModel**)*g_iRHeight);
    for (i = 0; i < g_iRHeight; i++) {
        g_ColorModel[i] = (ColorModel**)malloc(sizeof(ColorModel*)*g_iRWidth);
        for (j = 0; j < g_iRWidth; j++) {
            //initialization of each CodeBookArray.
            g_ColorModel[i][j] = (ColorModel*)malloc(sizeof(ColorModel));
            g_ColorModel[i][j]->m_Codewords = (ColorCodeword**)malloc(sizeof(ColorCodeword*)*iElementArraySize);
            g_ColorModel[i][j]->m_iNumEntries = 0;
            g_ColorModel[i][j]->m_iElementArraySize = iElementArraySize;
            g_ColorModel[i][j]->m_iTotal = 0;
            g_ColorModel[i][j]->m_bID = 1;
        }
    }

    //cache-book initialization
    if (g_bAbsorptionEnable == true) {
        iElementArraySize = 1;

        g_CCacheBook = (ColorModel***)malloc(sizeof(ColorModel**)*g_iRHeight);
        for (i = 0; i < g_iRHeight; i++) {
            g_CCacheBook[i] = (ColorModel**)malloc(sizeof(ColorModel*)*g_iRWidth);
            for (j = 0; j < g_iRWidth; j++) {
                //initialization of each CodeBookArray.
                g_CCacheBook[i][j] = (ColorModel*)malloc(sizeof(ColorModel));
                g_CCacheBook[i][j]->m_Codewords = (ColorCodeword**)malloc(sizeof(ColorCodeword*)*iElementArraySize);
                g_CCacheBook[i][j]->m_iNumEntries = 0;
                g_CCacheBook[i][j]->m_iElementArraySize = iElementArraySize;
                g_CCacheBook[i][j]->m_iTotal = 0;
                g_CCacheBook[i][j]->m_bID = 0;
            }
        }

        g_aCReferredIndex = (short**)malloc(sizeof(short*)*g_iRHeight);
        g_aCContinuousCnt = (short**)malloc(sizeof(short*)*g_iRHeight);
        for (i = 0; i < g_iRHeight; i++) {
            g_aCReferredIndex[i] = (short*)malloc(sizeof(short)*g_iRWidth);
            g_aCContinuousCnt[i] = (short*)malloc(sizeof(short)*g_iRWidth);
            for (j = 0; j < g_iRWidth; j++) {
                g_aCReferredIndex[i][j] = -1;
                g_aCContinuousCnt[i][j] = 0;
            }
        }
    }
}

//-----------------------------------------------------------------------------------------------------------------------------------------//
//															the memory release function											           //
//-----------------------------------------------------------------------------------------------------------------------------------------//
void MultiCues::C_ReleaseColorModelRelatedMemory() {
    int i, j, k;

    for (i = 0; i < g_iRHeight; i++) {
        for (j = 0; j < g_iRWidth; j++) {
            for (k = 0; k < g_ColorModel[i][j]->m_iNumEntries; k++) {
                free(g_ColorModel[i][j]->m_Codewords[k]);
            }
            free(g_ColorModel[i][j]->m_Codewords);
            free(g_ColorModel[i][j]);
        }
        free(g_ColorModel[i]);
    }
    free(g_ColorModel);

    if (g_bAbsorptionEnable == true) {
        for (i = 0; i < g_iRHeight; i++) {
            for (j = 0; j < g_iRWidth; j++) {
                for (k = 0; k < g_CCacheBook[i][j]->m_iNumEntries; k++) {
                    free(g_CCacheBook[i][j]->m_Codewords[k]);
                }
                free(g_CCacheBook[i][j]->m_Codewords);
                free(g_CCacheBook[i][j]);
            }
            free(g_CCacheBook[i]);
        }
        free(g_CCacheBook);

        for (i = 0; i < g_iRHeight; i++) {
            free(g_aCReferredIndex[i]);
            free(g_aCContinuousCnt[i]);
        }
        free(g_aCReferredIndex);
        free(g_aCContinuousCnt);
    }
}

//-----------------------------------------------------------------------------------------------------------------------------------------//
//														the codebook construction function								                   //
//-----------------------------------------------------------------------------------------------------------------------------------------//
void MultiCues::C_CodebookConstruction(uchar** aP, int iPosX, int iPosY, short nTrainVolRange, float fLearningRate, ColorModel* pC) {

    //Step1: matching
    short nMatchedIndex;

    float fNegLearningRate = 1 - fLearningRate;

    nMatchedIndex = -1;

    for (int i = 0; i < pC->m_iNumEntries; i++) {

        // Huong thay doi
        if(pC->m_Codewords[i]->m_dMean - nTrainVolRange <= aP[iPosY][iPosX] && aP[iPosY][iPosX] <= pC->m_Codewords[i]->m_dMean + nTrainVolRange) {
            nMatchedIndex = i;
            break;
        }
    }

    pC->m_iTotal = pC->m_iTotal + 1;

    //Step2 : adding a new element
    if (nMatchedIndex == -1) {
        if (pC->m_iElementArraySize == pC->m_iNumEntries) {
            pC->m_iElementArraySize = pC->m_iElementArraySize + 5;
            ColorCodeword **temp = (ColorCodeword**)malloc(sizeof(ColorCodeword*)*pC->m_iElementArraySize);
            for (int j = 0; j < pC->m_iNumEntries; j++) {
                temp[j] = pC->m_Codewords[j];
                pC->m_Codewords[j] = NULL;
            }
            free(pC->m_Codewords);
            pC->m_Codewords = temp;
        }
        pC->m_Codewords[pC->m_iNumEntries] = (ColorCodeword*)malloc(sizeof(ColorCodeword));
        // Huong thay doi

        pC->m_Codewords[pC->m_iNumEntries]->m_dMean = aP[iPosY][iPosX];

        pC->m_Codewords[pC->m_iNumEntries]->m_iT_first_time = pC->m_iTotal;
        pC->m_Codewords[pC->m_iNumEntries]->m_iT_last_time = pC->m_iTotal;
        pC->m_Codewords[pC->m_iNumEntries]->m_iMNRL = pC->m_iTotal - 1;
        pC->m_iNumEntries = pC->m_iNumEntries + 1;
    }

    //Step3 : update
    else {
        //m_dMean update

        pC->m_Codewords[nMatchedIndex]->m_dMean = fLearningRate *aP[iPosY][iPosX] + fNegLearningRate*pC->m_Codewords[nMatchedIndex]->m_dMean;
        pC->m_Codewords[nMatchedIndex]->m_iT_last_time = pC->m_iTotal;
    }

    //cache-book handling
    if (pC->m_bID == 1) {
        //1. m_iMNRL update
        int iNegTime;
        for (int i = 0; i < pC->m_iNumEntries; i++) {
            //m_iMNRL update
            iNegTime = pC->m_iTotal - pC->m_Codewords[i]->m_iT_last_time + pC->m_Codewords[i]->m_iT_first_time - 1;
            if (pC->m_Codewords[i]->m_iMNRL < iNegTime) pC->m_Codewords[i]->m_iMNRL = iNegTime;
        }

        //2. g_aCReferredIndex[iPosY][iPosX] update
        if (g_bAbsorptionEnable == true)
            g_aCReferredIndex[iPosY][iPosX] = -1;
    }

    else {
        //1. m_iMNRL update:
        if (nMatchedIndex == -1) pC->m_Codewords[pC->m_iNumEntries - 1]->m_iMNRL = 0;

        //2. g_aCReferredIndex[iPosY][iPosX] update
        if (nMatchedIndex == -1) {
            g_aCReferredIndex[iPosY][iPosX] = pC->m_iNumEntries - 1;
            g_aCContinuousCnt[iPosY][iPosX] = 1;
        }
        else {
            if (nMatchedIndex == g_aCReferredIndex[iPosY][iPosX])
                g_aCContinuousCnt[iPosY][iPosX]++;
            else {
                g_aCReferredIndex[iPosY][iPosX] = nMatchedIndex;
                g_aCContinuousCnt[iPosY][iPosX] = 1;
            }
        }
    }
}

//-----------------------------------------------------------------------------------------------------------------------------------------//
//												Clear non-essential codewords of the given codebook							               //																													   //
//-----------------------------------------------------------------------------------------------------------------------------------------//
void MultiCues::C_ClearNonEssentialEntries(short nClearNum, ColorModel* pModel) {
    int i;
    short nStaleThresh = (int)(nClearNum*0.5);
    short nKeepCnt;
    int* aKeep;

    ColorModel* pC = pModel;

    if (pC->m_iTotal < nClearNum) return; //(Being operated only when pC->t >= nClearNum)

    //Step1:initialization
    aKeep = (int*)malloc(sizeof(int)*pC->m_iNumEntries);

    nKeepCnt = 0;

    //Step2: Find non-essential codewords
    for (i = 0; i < pC->m_iNumEntries; i++) {
        if (pC->m_Codewords[i]->m_iMNRL > nStaleThresh) {
            aKeep[i] = 0; //removal
        }
        else {
            aKeep[i] = 1; //keep
            nKeepCnt++;
        }
    }

    //Step3: Perform removal
    if (nKeepCnt == 0 || nKeepCnt == pC->m_iNumEntries) {
        for (i = 0; i < pC->m_iNumEntries; i++) {
            pC->m_Codewords[i]->m_iT_first_time = 1;
            pC->m_Codewords[i]->m_iT_last_time = 1;
            pC->m_Codewords[i]->m_iMNRL = 0;
        }
    }
    else {
        nKeepCnt = 0;
        ColorCodeword** temp = (ColorCodeword**)malloc(sizeof(ColorCodeword*)*pC->m_iNumEntries);

        for (i = 0; i < pC->m_iNumEntries; i++) {
            if (aKeep[i] == 1) {
                temp[nKeepCnt] = pC->m_Codewords[i];
                temp[nKeepCnt]->m_iT_first_time = 1;
                temp[nKeepCnt]->m_iT_last_time = 1;
                temp[nKeepCnt]->m_iMNRL = 0;
                nKeepCnt++;
            }
            else free(pC->m_Codewords[i]);
        }

        //ending..
        free(pC->m_Codewords);
        pC->m_Codewords = temp;
        pC->m_iElementArraySize = pC->m_iNumEntries;
        pC->m_iNumEntries = nKeepCnt;
    }

    pC->m_iTotal = 0;
    free(aKeep);

}

//-----------------------------------------------------------------------------------------------------------------------------------------//
//										Clear non-essential codewords of the given codebook (for cache-book)				               //
//-----------------------------------------------------------------------------------------------------------------------------------------//
void MultiCues::C_ClearNonEssentialEntriesForCachebook(uchar bLandmark, short nReferredIdx, short nClearNum, ColorModel* pCachebook) {
    int i;

    if (pCachebook->m_iTotal < nClearNum) {
        for (i = 0; i < pCachebook->m_iNumEntries; i++) {
            if (bLandmark == 255 && i == nReferredIdx) pCachebook->m_Codewords[i]->m_iMNRL = 0;
            else pCachebook->m_Codewords[i]->m_iMNRL++;
        }

        pCachebook->m_iTotal++;
    }

    else {
        int iStaleThreshold = 5;

        int* aKeep;
        short nKeepCnt;

        aKeep = (int*)malloc(sizeof(int)*pCachebook->m_iNumEntries);
        nKeepCnt = 0;

        for (i = 0; i < pCachebook->m_iNumEntries; i++) {
            if (pCachebook->m_Codewords[i]->m_iMNRL < iStaleThreshold) {
                aKeep[i] = 1;
                nKeepCnt++;
            }
            else aKeep[i] = 0;
        }

        pCachebook->m_iElementArraySize = nKeepCnt + 2;
        //		if (pCachebook->m_iElementArraySize < 3) pCachebook->m_iElementArraySize = 3;

        ColorCodeword** temp = (ColorCodeword**)malloc(sizeof(ColorCodeword*)*pCachebook->m_iElementArraySize);
        nKeepCnt = 0;

        for (i = 0; i < pCachebook->m_iNumEntries; i++) {
            if (aKeep[i] == 1) {
                temp[nKeepCnt] = pCachebook->m_Codewords[i];
                temp[nKeepCnt]->m_iMNRL = 0;
                nKeepCnt++;
            }
            else {
                free(pCachebook->m_Codewords[i]);
            }

        }

        //ending..
        free(pCachebook->m_Codewords);
        pCachebook->m_Codewords = temp;
        pCachebook->m_iNumEntries = nKeepCnt;
        pCachebook->m_iTotal = 0;

        free(aKeep);
    }
}

//-----------------------------------------------------------------------------------------------------------------------------------------//
//														the ghost-region absorption function										       //
//-----------------------------------------------------------------------------------------------------------------------------------------//
void MultiCues::C_Absorption(int iAbsorbCnt, point pos, short** aContinuCnt, short** aRefferedIndex, ColorModel* pModel, ColorModel* pCache) {

    //set iLeavingIndex
    if (aContinuCnt[pos.m_nY][pos.m_nX] < iAbsorbCnt) return;

    int iLeavingIndex = aRefferedIndex[pos.m_nY][pos.m_nX];

    //array expansion
    if (pModel->m_iElementArraySize == pModel->m_iNumEntries) {
        pModel->m_iElementArraySize = pModel->m_iElementArraySize + 5;
        ColorCodeword** temp = (ColorCodeword**)malloc(sizeof(ColorCodeword*)*pModel->m_iElementArraySize);
        for (int i = 0; i < pModel->m_iNumEntries; i++) temp[i] = pModel->m_Codewords[i];
        free(pModel->m_Codewords);
        pModel->m_Codewords = temp;
    }

    // movement from the cache-book to the codebook
    pModel->m_Codewords[pModel->m_iNumEntries] = pCache->m_Codewords[iLeavingIndex];

    pModel->m_iTotal = pModel->m_iTotal + 1;

    pModel->m_Codewords[pModel->m_iNumEntries]->m_iT_first_time = pModel->m_iTotal;
    pModel->m_Codewords[pModel->m_iNumEntries]->m_iT_last_time = pModel->m_iTotal;
    pModel->m_Codewords[pModel->m_iNumEntries]->m_iMNRL = pModel->m_iTotal - 1;

    pModel->m_iNumEntries = pModel->m_iNumEntries + 1;

    int k = 0;
    ColorCodeword **pTempCache = (ColorCodeword**)malloc(sizeof(ColorCodeword*)*pCache->m_iElementArraySize);
    for (int i = 0; i < pCache->m_iNumEntries; i++) {
        if (i == iLeavingIndex) continue;
        else {
            pTempCache[k] = pCache->m_Codewords[i];
            k++;
        }
    }
    free(pCache->m_Codewords);
    pCache->m_Codewords = pTempCache;
    pCache->m_iNumEntries = k;
}
