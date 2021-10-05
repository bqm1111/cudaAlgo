#ifndef CPUMULTICUE_HPP
#define CPUMULTICUE_HPP


#include <opencv2/opencv.hpp>
#include <chrono>
#include "../define.h"
#define getMoment std::chrono::high_resolution_clock::now()
#define getTimeElapsed(mess, end, start) {std::cout << mess << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()/1000.0 << std::endl;}
#define getExeTime(mess, ans) {auto start = getMoment;ans;auto end = getMoment; getTimeElapsed(mess, end, start);}

//------------------------------------Structure Lists-------------------------------------//

namespace multiCues
{
typedef bool BOOL;
//ProbModel bgModel;
//struct cmdModel{
//    ProbModel model;
//};
struct point {
    short m_nX;
    short m_nY;
};

struct neighbor_pos {
    short m_nX;
    short m_nY;
};
//1) Bounding Box Structure
struct BoundingBoxInfo {
    int m_iBoundBoxNum;										//# of bounding boxes for all foreground and false-positive blobs
    int m_iArraySize;										//the size of the below arrays to store bounding box information

    short *m_aLeft, *m_aRight, *m_aUpper, *m_aBottom;		//arrays to store bounding box information for (the original frame size)
    short *m_aRLeft, *m_aRRight, *m_aRUpper, *m_aRBottom;	//arrays to store bounding box information for (the reduced frame size)


    BOOL* m_ValidBox;										//If this value is true, the corresponding bounding box is for a foreground blob.
    //Else, it is for a false-positive blob
};

//2) Texture Model Structure
struct TextureCodeword {
    int m_iMNRL;											//the maximum negative run-length
    int m_iT_first_time;									//the first access time
    int m_iT_last_time;										//the last access time

    float m_fLowThre;										//a low threshold for the matching
    float m_fHighThre;										//a high threshold for the matching
    float m_fMean;											//mean of the codeword
};

struct TextureModel {
    TextureCodeword** m_Codewords;							//the texture-codeword Array

    int m_iTotal;											//# of learned samples after the last clear process
    int m_iElementArraySize;								//the array size of m_Codewords
    int m_iNumEntries;										//# of codewords

    BOOL m_bID;												//id=1 --> background model, id=0 --> cachebook
};

//3) Color Model Structure
struct ColorCodeword {
    int m_iMNRL;											//the maximum negative run-length
    int m_iT_first_time;									//the first access time
    int m_iT_last_time;										//the last access time


    double m_dMean;

};

struct ColorModel {
    ColorCodeword** m_Codewords;							//the color-codeword Array

    int m_iTotal;											//# of learned samples after the last clear process
    int m_iElementArraySize;								//the array size of m_Codewords
    int m_iNumEntries;										//# of codewords

    BOOL m_bID;												//id=1 --> background model, id=0 --> cachebookk
};

struct GaussBG{
    float h_mean;
    float h_var;
};
}


namespace multiCues
{
class MultiCues
{
private:
    void save_config(cv::FileStorage &fs);
    void load_config(cv::FileStorage &fs);

public:
//	typedef bgslibrary::algorithms::multiCue::point point;
//	typedef bgslibrary::algorithms::multiCue::TextureModel TextureModel;
//	typedef bgslibrary::algorithms::multiCue::BoundingBoxInfo BoundingBoxInfo;
//	typedef bgslibrary::algorithms::multiCue::ColorModel ColorModel;
//	typedef bgslibrary::algorithms::multiCue::BOOL BOOL;

    MultiCues();
    ~MultiCues();
    //----------------------------------------------------
    //		APIs and User-Adjustable Parameters
    //----------------------------------------------------
    void process(const cv::Mat &img_input, cv::Mat &img_output, cv::Mat &img_bgmodel, cv::Mat& H);			//the main function to background modeling and subtraction

    void GetForegroundMap(IplImage* return_image, IplImage* input_frame = NULL);					//the function returning a foreground binary-map
    void Destroy();																				//the function to release allocated memories

    int g_iTrainingPeriod;										//the training period								(The parameter t in the paper)
    int g_iT_ModelThreshold;									//the threshold for texture-model based BGS.		(The parameter tau_T in the paper)
    int g_iC_ModelThreshold;									//the threshold for appearance based verification.  (The parameter tau_A in the paper)

    float g_fLearningRate;										//the learning rate for background models.			(The parameter alpha in the paper)

    short g_nTextureTrainVolRange;								//the codebook size factor for texture models.		(The parameter k in the paper)
    short g_nColorTrainVolRange;								//the codebook size factor for color models.		(The parameter eta_1 in the paper)

    //----------------------------------------------------
    //	Implemented Function Lists
    //----------------------------------------------------

    //--0) inherited from IGBS.h
    bool firstTime = true;
    bool showOutput = true;
    cv::Mat img_background;
    cv::Mat img_foreground;
    cv::Mat h_Homography;
//    double h[9];
//    LKTTracker lkt;

    void init(const cv::Mat &img_input, cv::Mat &img_outfg, cv::Mat &img_outbg);

    //--1) General Functions
    void Initialize(IplImage* frame);

    void PreProcessing(IplImage* frame);
    void ReduceImageSize(IplImage* SrcImage, IplImage* DstImage);
    void GaussianFiltering(IplImage* frame, uchar*** aFilteredFrame);
    void BGR2HSVxyz_Par(uchar*** aBGR, uchar*** aXYZ);

    void h_RGB2GRAY_Par(IplImage* frame, uchar** gray);
    void BackgroundModeling_Par(IplImage* frame);
    void ForegroundExtraction(IplImage* frame);
//	void CreateLandmarkArray_Par(float fConfThre, short nTrainVolRange, float**aConfMap, int iNehborNum, uchar*** aXYZ,
//								 point*** aNeiDir, TextureModel**** TModel, ColorModel*** CModel, uchar**aLandmarkArr);
    // Huong thay doi
    void CreateLandmarkArray_Par(float fConfThre, short nTrainVolRange, float**aConfMap, int iNehborNum, uchar*** aXYZ, uchar** gray,
                                 point*** aNeiDir, TextureModel**** TModel, ColorModel*** CModel, uchar**aLandmarkArr);

    void PostProcessing(IplImage* frame);
    void MorphologicalOpearions(uchar** aInput, uchar** aOutput, double dThresholdRatio, int iMaskSize, int iWidth, int iHeight);
    void Labeling(uchar** aBinaryArray, int* pLabelCount, int** aLabelTable);
    void SetBoundingBox(int iLabelCount, int** aLabelTable);
    void BoundBoxVerification(IplImage* frame, uchar** aResForeMap, BoundingBoxInfo* BoundBoxInfo);
    void EvaluateBoxSize(BoundingBoxInfo* BoundBoxInfo);
    void EvaluateOverlapRegionSize(BoundingBoxInfo* SrcBoxInfo);
    void EvaluateGhostRegion(IplImage* frame, uchar** aResForeMap, BoundingBoxInfo* BoundBoxInfo);
    // Huong thay doi
    void h_EvaluateGhostRegion(IplImage* frame, uchar** aResForeMap, BoundingBoxInfo* BoundBoxInfo);

    double CalculateHausdorffDist(IplImage* input_image, IplImage* model_image);
    void RemovingInvalidForeRegions(uchar** aResForeMap, BoundingBoxInfo* BoundBoxInfo);

    void UpdateModel_Par();
    void GetEnlargedMap(float** aOriginMap, float** aEnlargedMap);

    //--2) Texture Model Related Functions
    void T_AllocateTextureModelRelatedMemory();
    void T_ReleaseTextureModelRelatedMemory();
    void T_SetNeighborDirection(point*** aNeighborPos);
    //Huong add
    void T_ModelConstruction(short nTrainVolRange, float fLearningRate, uchar** gray, point center, point* aNei, TextureModel** aModel);
    void T_ClearNonEssentialEntries(short nClearNum, TextureModel** aModel);
    void T_ClearNonEssentialEntriesForCachebook(uchar bLandmark, short* nReferredIdxArr, short nClearNum, TextureModel** pCachebook);
    // Huong add
    void T_GetConfidenceMap_Par(uchar** gray, float** aTextureMap, point*** aNeiDirArr, TextureModel**** aModel);
    void T_Absorption(int iAbsorbCnt, point pos, short*** aContinuCnt, short*** aRefferedIndex, TextureModel** pModel, TextureModel** pCache);

    //--3) Color Model Related Functions
    void C_AllocateColorModelRelatedMemory();
    void C_ReleaseColorModelRelatedMemory();
    void C_CodebookConstruction(uchar** aP, int iPosX, int iPosY, short nTrainVolRange, float fLearningRate, ColorModel* pC);
    void C_ClearNonEssentialEntries(short nClearNum, ColorModel* pModel);
    void C_ClearNonEssentialEntriesForCachebook(uchar bLandmark, short nReferredIdx, short nClearNum, ColorModel* pCachebook);
    void C_Absorption(int iAbsorbCnt, point pos, short** aContinuCnt, short** aRefferedIndex, ColorModel* pModel, ColorModel* pCache);

    //--4) Motion background model estimation
    void h_MotionCompensation(cv::Mat H, TextureModel**** aModel, TextureModel**** hMode, ColorModel*** pC);
    void h_MatchingCodewords(TextureModel** aModel, TextureModel** dstModel);
    void h_TextureMotion(cv::Mat H,  float**aConfMap);

   //--5) Gauss background modeling
    void h_CreatGaussBG(GaussBG*** bgModel, uchar** aP, float h_learningrate, BOOL**  updateMap);
    void h_UpdateGaussBG(IplImage* frame, GaussBG*** gModel,  BoundingBoxInfo* BoundBoxInfo,  uchar** gray, BOOL** updatemap, uchar** aResForeMap);
    void h_ImAdjust(IplImage* scr, IplImage* dst, cv::Size sz) ; //, int tol = 1, cv::Vec2i in = cv::Vec2i(0, 255), cv::Vec2i out = cv::Vec2i(0, 255));
    void h_Stitching(const cv::Mat1b& src, cv::Mat1b& dst, int tol = 1, cv::Vec2i in = cv::Vec2i(0, 255), cv::Vec2i out = cv::Vec2i(0, 255));
    void h_CaclOverlap(BoundingBoxInfo* BoundBoxInfo);
    void h_AllocateGaussModelRelatedMemory();
    void h_ReleaseGaussModelRelatedMemory();

    //----------------------------------------------------
    //	Implemented Variable Lists
    //----------------------------------------------------

    //--1) General Variables
    int g_iFrameCount;							//the counter of processed frames

    int g_iBackClearPeriod;						//the period to clear background models
    int g_iCacheClearPeriod;					//the period to clear cache-book models

    int g_iAbsortionPeriod;						//the period to absorb static ghost regions
    BOOL g_bAbsorptionEnable;					//If True, procedures for ghost region absorption are activated.

    BOOL g_bModelMemAllocated;					//To handle memory..
    BOOL g_bNonModelMemAllocated;				//To handle memory..

    float g_fConfidenceThre;					//the final decision threshold

    int g_iWidth, g_iHeight;					//width and height of input frames
    int g_iRWidth, g_iRHeight;					//width and height of reduced frames (For efficiency, the reduced size of frames are processed)
    int g_iForegroundNum;						//# of detected foreground regions
    BOOL g_bForegroundMapEnable;				//TRUE only when BGS is successful

    IplImage* g_ResizedFrame;					//reduced size of frame (For efficiency, the reduced size of frames are processed)
    uchar** h_GrayFrame;
    IplImage* h_mGrayFrame;
    IplImage* h_DiffFrame;

    uchar*** g_aGaussFilteredFrame;
    uchar*** g_aXYZFrame;
    uchar** g_aLandmarkArray;					//the landmark map
    uchar** g_aResizedForeMap;					//the resized foreground map
    uchar** g_aForegroundMap;					//the final foreground map
    BOOL** g_aUpdateMap;						//the location map of update candidate pixels

    BoundingBoxInfo* g_BoundBoxInfo;			//the array of bounding boxes of each foreground blob

    //--2) Texture Model Related
    TextureModel**** g_TextureModel;			//the texture background model
    TextureModel**** g_TCacheBook;				//the texture cache-book
    TextureModel**** h_TextureModel;
    short*** g_aTReferredIndex;					//To handle cache-book
    short*** g_aTContinuousCnt;					//To handle cache-book
    point*** g_aNeighborDirection;
    float**g_aTextureConfMap;					//the texture confidence map

    short g_nNeighborNum;						//# of neighborhoods
    short g_nRadius;
    short g_nBoundarySize;

    //--3) Texture Model Related
    ColorModel*** g_ColorModel;					//the color background model
    ColorModel*** g_CCacheBook;					//the color cache-book
    short** g_aCReferredIndex;					//To handle cache-book
    short** g_aCContinuousCnt;					//To handle cache-book
    //--4) Gauss Model
    float h_blocksize;
    int h_modelW;
    int h_modelH;
    float THRES_BG_REFINE;
//    float h_learningrate;
//    IplImage* checkdisplay;
    bool h_init;
    bool h_done;

    GaussBG*** h_BGModel;
    GaussBG*** h_tmpModel;

};
}




#endif // CPUMULTICUE_HPP
