/*
 * Written by Quang Minh Bui(minhbq6@viettel.com.vn, buiquangminh1111@gmail.com)
 *
/*
    This software library implements CUDA version of the background subtraction algorithm
    described in

        "A New Framework for Background Subtraction Using Multiple Cues"
        written by SeungJong Noh and Moongu Jeon.
        In Asian Conference on Computer Vision: ACCV 2012

    ----------------------------------------------------------------------
*/
#ifndef MULTICUE_HPP
#define MULTICUE_HPP
#include "../define.h"
#include "common_structs.hpp"
#include "../utils.hpp"
#define GAUSS
#define MIN_AREA 1500
#define MAX_AREA 80000

namespace multiCue {
class MultiCues
{
#define CUDA_MAX_CODEWORDS_SIZE 20
#define PRINTTOCHECK
public:
    unsigned char * m_dsrcImg;           // Input image. Expected to be gray image
    unsigned char * m_dResizedImg;       // reduced size of frame (for efficiency)
    unsigned char * m_dFilteredImg;        // Final image to process after gaussian blurring
    unsigned char * m_hFilteredImg;
    unsigned char * m_dfgMap;            // foreground Map
    unsigned char * m_dbgMap;            // background Map
    unsigned char * m_dlandmarkMap;      // Landmark Map
    unsigned char * m_dResizedFgMap;     // the resized foreground Map
    unsigned char * m_hResizedFgMap;
    cv::Mat m_fgMap;

    bool * m_dUpdateMap;                 // the location map of update candidate pixel (matrix of boolean value)
    bool * m_hUpdateMap;
    bool * m_dUpdateMapCache;            // the location map of update candidate pixel

    BoundingBoxInfo* m_BboxInfo;			//the array of bounding boxes of each foreground blob

    // Texture Model Related Variables
    TextureCodeword* m_TCodeword;
    TextureModel *m_TextureModel;        // the texture background model
    TextureCodeword * m_codeWordTCacheBook;
    TextureModel *m_TCacheBook;          // the texture cache-book

    short * m_TReferredIndex;             // to handle cache book
    short * m_TContinuousCnt;              // to handle cache book

    short2 * m_neighborDirection;
    float * m_textureConfMap;

    int * m_TkeepCnt;
    TextureCodeword * m_TCodewordTemp;
    TextureCodeword * m_TCodewordTempCache;

    // Color Model Related
    ColorCodeword * m_CCodeword;
    ColorModel *m_ColorModel;
    ColorCodeword * m_codeWordCCacheBook;
    ColorModel *m_CCacheBook;
    short * m_CReferredIndex;
    short * m_CContinuousCnt;
    int * m_CkeepCnt;
    ColorCodeword * m_CCodewordTemp;
    ColorCodeword * m_CCodewordTempCache;


    // Gauss Model
    GaussModel * m_dGaussianBgModel;
    GaussModel * m_hGaussianBgModel;

    float m_gaussBlockSize;
    int m_gaussModelW;
    int m_gaussModelH;
    float THRES_BG_REFINE;

    bool h_init;
    bool h_done;

    short m_neighborNum;						//# of neighborhoods
    short m_neighborRadius;
    short m_boundarySize;


public:
    MultiCues();
    ~MultiCues();

    std::vector<cv::Rect> m_objRect;

    // Main function
    void movingDetectObject(unsigned char * d_srcImg, unsigned char * h_srcImg, int srcWidth, int srcHeight);
    void initParam(unsigned char * d_srcImg, int imgWidth, int imgHeight);
    void process(unsigned char * d_img, unsigned char * h_img);

    void bgModelling(unsigned char * d_img);
    void fgExtraction(unsigned char * d_img, unsigned char * h_img);
    void updateModel();
    void getFgMap();

    void preProcessing(unsigned char * d_img);
    void releaseMem();



public:
    // CUDA Helper function
    // Init function
    void allocateTextureModelRelatedMemory();
    void allocateTextureModelRelatedMemoryHelper(TextureModel * TModel, int iElementArraySize, int _bID,
                                                 int rWidth, int rHeight, int neighborNum);
    void allocateGaussModelRelatedMemory();
    void allocateColorModelRelatedMemoryHelper(ColorModel * CModel, int iElementArraySize, int _bID,
                                                 int rWidth, int rHeight);

    void allocateColorModelRelatedMemory();

    void releaseTextureModelRelatedMemory();
    void releaseGaussModelRelatedMemory();
    void releaseColorModelRelatedMemory();

    void setNeighborDirection();

    // Pre - Processing function
    void gpuResize(const unsigned char * d_src, unsigned char * d_dst,
                   int srcWidth, int srcHeight,
                   int dstWidth, int dstHeight);
    void gpuGaussianBlur(const unsigned char * d_src, unsigned char * d_dst,
                         int width, int height, double sigma = 0.7);

    // Background Modelling function
    void gpuCreateGaussianModel(float learningRate, bool * mask = nullptr);
    void gpuTextureModelConstruction(short nTrainVolRange, float learningRate,
                                     unsigned char *gray, short2 * neighborDirection,
                                     TextureModel *TModel,
                                     TextureCodeword * TCodeWord,
                                     short * TReferredIndex,
                                     short * TContinuousCnt,
                                     int radius,
                                     int rWidth, int rHeight, int neighborNum, bool * mask = nullptr);

    void gpuColorCodeBookConstruction(short nTrainVolRange, float learningRate,
                                      unsigned char * gray, ColorModel * CModel,
                                      ColorCodeword * CCodeword,
                                      short * CReferredIndex,
                                      short * CContinuousCnt,
                                      int radius,
                                      int rWidth, int rHeight,
                                      bool * mask = nullptr);
    void gpuTextureClearNonEssentialEntries(short nClearNum, TextureModel *aModel,
                                            TextureCodeword * aCodeword, bool * mask = nullptr);
    void gpuColorClearNonEssentialEntries(short nClearNum, ColorModel * aModel,
                                          ColorCodeword * aCodeword, bool * mask = nullptr);

    // foreground extraction function
    void gpuGetConfidenceMap(unsigned char * gray, float *aTextureMap, short2 * neighborDir, TextureModel * TModel, TextureCodeword * TCodeword);

    void gpuCreateLandMarkArray(float fConfThresh, short nTrainVolRange, float *aConfMap, unsigned char * gray,
                                short2 * neighborDir, TextureModel *TModel, ColorModel * CModel, TextureCodeword * TCodeword,
                                ColorCodeword * CCodeword, unsigned char * landMarkMap);
    void gpuPostProcessing(unsigned char * h_img);
    void gpuMorphologicalOperation(unsigned char * src, unsigned char *dst, float thresholdRatio, int maskSize, int rWidth, int rHeight);
    void getBoundingBox();
    void gpuLabelling(unsigned char * src, int *labelCnt, int *labelTable);

    void boundingBoxVerification(unsigned char * h_img);
    void evaluateBoxSize();
    void evaluateGhostRegion(unsigned char * h_img);
    void calcOverlap();
    void gaussianRefineBgModel(unsigned char * h_img);
    void removingInvalidFgRegion();
    double CalculateHausdorffDist(IplImage* input_image, IplImage* model_image);
    // Update model function
    void gpuUpdateModel(BoundingBoxInfo * bboxInfo, unsigned char * d_ResizeFgMap, bool * updateMap, bool * updateMapCache);
    void gpuTextureAbsorption(int absorpCnt, short * TContinuousCnt, short * TReferredIndex,
                              TextureModel *TModel, TextureCodeword * TCodeword,
                              TextureModel *TCacheBook, TextureCodeword * TCodewordCacheBook, bool * mask = nullptr);

    void gpuColorAbsorption(int absorpCnt, short * CContinuousCnt, short* CReferredIndex,
                            ColorModel * CModel, ColorCodeword * CCodeword,
                            ColorModel * CCacheBook, ColorCodeword * CCodewordCacheBook, bool * mask = nullptr);

    void gpuTextureClearNonEssentialEntriesForCacheBook(short nClearNum,
                                                        unsigned char * landmarkMap,
                                                        short * TReferredIdx,
                                                        TextureModel * TCacheBook,
                                                        TextureCodeword * TCodeword);
    void gpuColorClearNonEssentialEntriesForCacheBook(short nClearNum,
                                                      unsigned char * landmarkMap,
                                                      short * CRefferedIdx,
                                                      ColorModel * CCacheBook,
                                                      ColorCodeword * CCodeword);
    int m_trainingPeriod;										//the training period								(The parameter t in the paper)
    int g_iT_ModelThreshold;									//the threshold for texture-model based BGS.		(The parameter tau_T in the paper)
    int g_iC_ModelThreshold;									//the threshold for appearance based verification.  (The parameter tau_A in the paper)

    float m_learningRate;										//the learning rate for background models.			(The parameter alpha in the paper)

    short m_nTextureTrainVolRange;								//the codebook size factor for texture models.		(The parameter k in the paper)
    short m_nColorTrainVolRange;								//the codebook size factor for color models.		(The parameter eta_1 in the paper)

    //----------------------------------------------------
    //	Implemented Function Lists
    //----------------------------------------------------

    //--0) inherited from IGBS.h
    bool firstTime = true;
    bool showOutput = true;
    cv::Mat img_background;
    cv::Mat img_foreground;
    cv::Mat h_Homography;

    //----------------------------------------------------
    //	Implemented Variable Lists
    //----------------------------------------------------

    //--1) General Variables
    int m_frameCount;							//the counter of processed frames

    int m_backClearPeriod;						//the period to clear background models
    int g_iCacheClearPeriod;					//the period to clear cache-book models

    int m_AbsorptionPeriod;						//the period to absorb static ghost regions
    bool m_AbsorptionEnable;					//If True, procedures for ghost region absorption are activated.

    bool m_ModelMemAllocated;					//To handle memory..
    bool m_NonModelMemAllocated;				//To handle memory..

    float m_confidenceThresh;					//the final decision threshold

    int m_width, m_height;					//width and height of input frames
    int m_ResizeWidth, m_ResizeHeight;					//width and height of reduced frames (For efficiency, the reduced size of frames are processed)
    int m_fgNum;						//# of detected foreground regions
    bool m_fgMapEnable;                         //TRUE only when BGS is successful

};
}
#endif // MULTICUE_HPP
