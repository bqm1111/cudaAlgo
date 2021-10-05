#ifndef COMMON_STRUCTS_HPP
#define COMMON_STRUCTS_HPP
#include <vector_types.h>
namespace multiCue {
struct BoundingBoxInfo{
    int boxNum;
    short4 * dbox;
    short4 * dRbox;

    short4 * hbox;
    short4 * hRbox;
    bool * d_isValidBox;
    bool * h_isValidBox;
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
    int m_CodewordsIdx;										//the texture-codeword Array

    int m_iTotal;											//# of learned samples after the last clear process
    int m_iElementArraySize;								//the array size of m_Codewords
    int m_iNumEntries;										//# of codewords

    bool m_bID;												//id=1 --> background model, id=0 --> cachebook
};

struct ColorCodeword {
    int m_iMNRL;											//the maximum negative run-length
    int m_iT_first_time;									//the first access time
    int m_iT_last_time;										//the last access time

    float m_dMean;

};

struct ColorModel {
    int m_CodewordsIdx;         							//the color-codeword Array

    int m_iTotal;											//# of learned samples after the last clear process
    int m_iElementArraySize;								//the array size of m_Codewords
    int m_iNumEntries;										//# of codewords

    bool m_bID;												//id=1 --> background model, id=0 --> cachebookk
};

struct GaussModel
{
    float mean;
    float var;
};
}
#endif // COMMON_STRUCTS_HPP
