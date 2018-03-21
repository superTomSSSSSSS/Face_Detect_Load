#pragma once
#include<opencv2\opencv.hpp>
#include<thread>

#define MAXFACENUM 20

class VideoFaceDetector
{
public:
	VideoFaceDetector(const std::string cascadeFilePath, const std::string cascadeFilePathcopy, cv::VideoCapture &videoCapture);
    ~VideoFaceDetector();

	std::vector<cv::Point>  getFrameAndDetect(cv::Mat &frame);
	std::vector<cv::Point>  operator>>(cv::Mat &frame);
    void                    setVideoCapture(cv::VideoCapture &videoCapture);
    cv::VideoCapture*       videoCapture() const;
	void                    setFaceCascade(const std::string cascadeFilePath, const std::string cascadeFilePathcopy);
    cv::CascadeClassifier*  faceCascade() const;
    void                    setResizedWidth(const int width);
    int                     resizedWidth() const;
	bool					isFaceFound() const;
    cv::Rect                face(int i) const;
	cv::Point               facePosition(int i) const;
    void                    setTemplateMatchingMaxDuration(const double s);
    double                  templateMatchingMaxDuration() const;
	int                     m_face_num = -1;
	int                     m_jump_allSearch = 0;
	std::vector<int>        m_all_roi;

private:
    static const double     TICK_FREQUENCY;

    cv::VideoCapture*       m_videoCapture = NULL;
    cv::CascadeClassifier*  m_faceCascade = NULL;
	cv::CascadeClassifier*  m_faceCascadethread = NULL;
	std::vector<cv::Rect>   m_allFaces, m_trackedFaces, m_faceRois, m_roitrackedFace;
	std::vector<cv::Point>  m_facePositions;
	std::vector<cv::Mat>    m_faceTemplates;
    cv::Mat                 m_matchingResult;
    bool                    m_templateMatchingRunning = false;
	std::vector<int64>      m_templateMatchingStartTime;
    std::vector<int64>      m_templateMatchingCurrentTime;
	int64                   m_allsearchFaceStartTime = 0;
	int64                   m_allsearchFaceCurrentTime = 0;
    bool                    m_foundFace = false;
    double                  m_scale;
    int                     m_resizedWidth = 320;
    double                  m_templateMatchingMaxDuration = 3;  
	double                  m_allsearchFaceMaxDuration = 3;


    cv::Rect    doubleRectSize(const cv::Rect &inputRect, const cv::Rect &frameSize) const;
    cv::Rect    biggestFace(std::vector<cv::Rect> &faces) const;
	void        getbiggerTwoFace(std::vector<cv::Rect> &faces) const;
    cv::Point   centerOfRect(const cv::Rect &rect) const;
    cv::Mat     getFaceTemplate(const cv::Mat &frame, cv::Rect face);
    void        detectFaceAllSizes(const cv::Mat &frame);
	void        detectFaceAroundRoi(const cv::Mat &frame, int faceth);
    void        detectFacesTemplateMatching(const cv::Mat &frame, int faceth);
	void        Thread_All_Search(cv::Mat resizedFrame, int &result);
	void        Delete_Contain(std::vector<cv::Rect>&Input_Rects, std::vector<cv::Point>&Input_Points, int &facenum);
};

