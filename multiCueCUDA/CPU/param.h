#ifndef PARAM_H
#define PARAM_H


//----------------------------------
//	User adjustable parameters
//----------------------------------
int g_iTrainingPeriod = 5;											//the training period								(The parameter t in the paper)
int g_iT_ModelThreshold = 1;										//the threshold for texture-model based BGS.		(The parameter tau_T in the paper)
int g_iC_ModelThreshold = 10;										//the threshold for appearance based verification.  (The parameter tau_A in the paper)

float g_fLearningRate = 0.01f;											//the learning rate for background models.			(The parameter alpha in the paper)

short g_nTextureTrainVolRange = 5;									//the codebook size factor for texture models.		(The parameter k in the paper)
short g_nColorTrainVolRange = 20;										//the codebook size factor for color models.		(The parameter eta_1 in the paper)

bool g_bAbsorptionEnable = true;										//If true, cache-book is also modeled for ghost region removal.
int g_iAbsortionPeriod = 200;										//the period to absorb static ghost regions

int g_iRWidth = 160, g_iRHeight = 120;								//Frames are precessed after reduced in this size .

//------------------------------------
//	For codebook maintenance
//------------------------------------
int g_iBackClearPeriod = 200;		//300								//the period to clear background models
int g_iCacheClearPeriod = 30;		//30								//the period to clear cache-book models

//------------------------------------
//	Initialization of other parameters
//------------------------------------
int g_nNeighborNum = 6, g_nRadius = 2;
float g_fConfidenceThre = g_iT_ModelThreshold / (float)g_nNeighborNum;	//the final decision threshold

int g_iFrameCount = 0;
bool g_bForegroundMapEnable = false;									//true only when BGS is successful
bool g_bModelMemAllocated = false;									//To handle memory..
bool g_bNonModelMemAllocated = false;								//To handle memory..

int h_blocksize = 2;
int h_init = false;
int THRES_BG_REFINE =30;

//    h_learningrate = 0.05;

//  initLoadSaveConfig(algorithmName);

#endif // PARAM_H
