#include "VideoFaceDetector.h"
#include <iostream>
//#include <opencv2\imgproc.hpp>

const double VideoFaceDetector::TICK_FREQUENCY = cv::getTickFrequency();

VideoFaceDetector::VideoFaceDetector(const std::string cascadeFilePath, const std::string cascadeFilePathcopy, cv::VideoCapture &videoCapture)
{
	setFaceCascade(cascadeFilePath, cascadeFilePathcopy);
    setVideoCapture(videoCapture);
	m_trackedFaces.resize(MAXFACENUM);
	m_faceRois.resize(MAXFACENUM);
	m_facePositions.resize(MAXFACENUM);
	m_faceTemplates.resize(MAXFACENUM);
	m_roitrackedFace.resize(MAXFACENUM);
	m_templateMatchingStartTime.resize(MAXFACENUM);
	m_all_roi.resize(MAXFACENUM);
	m_templateMatchingCurrentTime.resize(MAXFACENUM);
	for (int i = 0; i < MAXFACENUM; i++)
	{
		m_templateMatchingStartTime.push_back(0);
		m_templateMatchingCurrentTime.push_back(0);
	}
	//m_matchingResults.resize(MAXFACENUM);
}

void VideoFaceDetector::setVideoCapture(cv::VideoCapture &videoCapture)
{
    m_videoCapture = &videoCapture;
}

cv::VideoCapture *VideoFaceDetector::videoCapture() const
{
    return m_videoCapture;
}

void VideoFaceDetector::setFaceCascade(const std::string cascadeFilePath, const std::string cascadeFilePathcopy)
{
    if (m_faceCascade == NULL) {
        m_faceCascade = new cv::CascadeClassifier(cascadeFilePath);
    }
    else {
        m_faceCascade->load(cascadeFilePath);
    }

	if (m_faceCascadethread == NULL) {
		m_faceCascadethread = new cv::CascadeClassifier(cascadeFilePathcopy);
	}
	else {
		m_faceCascadethread->load(cascadeFilePathcopy);
	}

	if (m_faceCascade->empty() || m_faceCascadethread->empty()) {
        std::cerr << "Error creating cascade classifier. Make sure the file" << std::endl
            << cascadeFilePath << " exists." << std::endl;
    }
}

cv::CascadeClassifier *VideoFaceDetector::faceCascade() const
{
    return m_faceCascade;
}

void VideoFaceDetector::setResizedWidth(const int width)
{
    m_resizedWidth = std::max(width, 1);
}

int VideoFaceDetector::resizedWidth() const
{
    return m_resizedWidth;
}

bool VideoFaceDetector::isFaceFound() const
{
	return m_foundFace;
}

cv::Rect VideoFaceDetector::face(int i) const
{
	cv::Rect faceRect;
	if (m_all_roi[i] == 1 || m_all_roi[i] == 3){
		faceRect = m_trackedFaces[i];
		faceRect.x = (int)(faceRect.x / m_scale);
		faceRect.y = (int)(faceRect.y / m_scale);
		faceRect.width = (int)(faceRect.width / m_scale);
		faceRect.height = (int)(faceRect.height / m_scale);
	}
	else if (m_all_roi[i] == 2)
	{
		faceRect = m_roitrackedFace[i];
		faceRect.x = (int)(faceRect.x / m_scale);
		faceRect.y = (int)(faceRect.y / m_scale);
		faceRect.width = (int)(faceRect.width / m_scale);
		faceRect.height = (int)(faceRect.height / m_scale);
	}
    return faceRect;
}

cv::Point VideoFaceDetector::facePosition(int i) const
{
    cv::Point facePos;
    facePos.x = (int)(m_facePositions[i].x / m_scale);
    facePos.y = (int)(m_facePositions[i].y / m_scale);
    return facePos;
}

void VideoFaceDetector::setTemplateMatchingMaxDuration(const double s)
{
    m_templateMatchingMaxDuration = s;
}

double VideoFaceDetector::templateMatchingMaxDuration() const
{
    return m_templateMatchingMaxDuration;
}

VideoFaceDetector::~VideoFaceDetector()
{
    if (m_faceCascade != NULL) {
        delete m_faceCascade;
    }
}

cv::Rect VideoFaceDetector::doubleRectSize(const cv::Rect &inputRect, const cv::Rect &frameSize) const
{
    cv::Rect outputRect;
    // Double rect size
    outputRect.width = inputRect.width * 2;
    outputRect.height = inputRect.height * 2;

    // Center rect around original center
    outputRect.x = inputRect.x - inputRect.width / 2;
    outputRect.y = inputRect.y - inputRect.height / 2;

    // Handle edge cases
    if (outputRect.x < frameSize.x) {
        outputRect.width += outputRect.x;
        outputRect.x = frameSize.x;
    }
    if (outputRect.y < frameSize.y) {
        outputRect.height += outputRect.y;
        outputRect.y = frameSize.y;
    }

    if (outputRect.x + outputRect.width > frameSize.width) {
        outputRect.width = frameSize.width - outputRect.x;
    }
    if (outputRect.y + outputRect.height > frameSize.height) {
        outputRect.height = frameSize.height - outputRect.y;
    }

    return outputRect;
}

cv::Point VideoFaceDetector::centerOfRect(const cv::Rect &rect) const
{
    return cv::Point(rect.x + rect.width / 2, rect.y + rect.height / 2);
}

cv::Rect VideoFaceDetector::biggestFace(std::vector<cv::Rect> &faces) const
{
    assert(!faces.empty());

    cv::Rect *biggest = &faces[0];
    for (auto &face : faces) {
        if (face.area() < biggest->area())
            biggest = &face;
    }
    return *biggest;
}

void VideoFaceDetector::getbiggerTwoFace(std::vector<cv::Rect> &faces) const
{
	assert(!faces.empty());

	cv::Rect temp;

	if (faces.size() > 1){
		for (int i = 0; i < faces.size(); i++)
		{
			for (int j = 0; j < faces.size() - i - 1; j++)
			{
				if (faces[j].area() < faces[j + 1].area())
				{
					temp = faces[j];
					faces[j] = faces[j + 1];
					faces[j + 1] = temp;
				}
			}
		}
	}
}

/*
* Face template is small patch in the middle of detected face.
*/
cv::Mat VideoFaceDetector::getFaceTemplate(const cv::Mat &frame, cv::Rect face)
{
    face.x += face.width / 4;
    face.y += face.height / 4;
    face.width /= 2;
    face.height /= 2;

    cv::Mat faceTemplate = frame(face).clone();
    return faceTemplate;
}

void VideoFaceDetector::detectFaceAllSizes(const cv::Mat &frame)
{
    // Minimum face size is 1/5th of screen height
    // Maximum face size is 2/3rds of screen height

	//进入全局人脸个数为0
	m_face_num = 0;

    m_faceCascade->detectMultiScale(frame, m_allFaces, 1.1, 3, 0,
        cv::Size(frame.rows / 5, frame.rows / 5),
        cv::Size(frame.rows * 2 / 3, frame.rows * 2 / 3));

    if (m_allFaces.empty()) return;

    m_foundFace = true;

    // Locate biggest face
	getbiggerTwoFace(m_allFaces);

	for (int i = 0; i < m_allFaces.size(); i++){

		m_trackedFaces[i] = m_allFaces[i];

		// Copy face template
		m_faceTemplates[i] = getFaceTemplate(frame, m_trackedFaces[i]);

		// Calculate roi
		m_faceRois[i] = doubleRectSize(m_trackedFaces[i], cv::Rect(0, 0, frame.cols, frame.rows));

		// Update face position
		m_facePositions[i] = centerOfRect(m_trackedFaces[i]);
	}
	m_face_num = m_allFaces.size();

	for (int ii = 0; ii < m_face_num; ii++)
		m_all_roi[ii] = 1;
}

void VideoFaceDetector::detectFaceAroundRoi(const cv::Mat &frame, int faceth)
{
	//进入roi模式
	m_all_roi[faceth] = 2;

    // Detect faces sized +/-20% off biggest face in previous search
	m_faceCascade->detectMultiScale(frame(m_faceRois[faceth]), m_allFaces, 1.1, 3, 0,
		cv::Size(m_trackedFaces[faceth].width * 8 / 10, m_trackedFaces[faceth].height * 8 / 10),
		cv::Size(m_trackedFaces[faceth].width * 12 / 10, m_trackedFaces[faceth].width * 12 / 10));

    if (m_allFaces.empty())
    {
        // Activate template matching if not already started and start timer
        m_templateMatchingRunning = true;
        if (m_templateMatchingStartTime[faceth] == 0)
            m_templateMatchingStartTime[faceth] = cv::getTickCount();
        return;
    }

    // Turn off template matching if running and reset timer
    m_templateMatchingRunning = false;
    m_templateMatchingCurrentTime[faceth] = m_templateMatchingStartTime[faceth] = 0;

    // Get detected face
    m_roitrackedFace[faceth] = biggestFace(m_allFaces);

    // Add roi offset to face
	m_roitrackedFace[faceth].x += m_faceRois[faceth].x;
	m_roitrackedFace[faceth].y += m_faceRois[faceth].y;

    // Get face template
	m_faceTemplates[faceth] = getFaceTemplate(frame, m_roitrackedFace[faceth]);

    // Calculate roi
	m_faceRois[faceth] = doubleRectSize(m_roitrackedFace[faceth], cv::Rect(0, 0, frame.cols, frame.rows));

    // Update face position
	m_facePositions[faceth] = centerOfRect(m_roitrackedFace[faceth]);
}

void VideoFaceDetector::detectFacesTemplateMatching(const cv::Mat &frame, int faceth)
{
	//进入模板匹配
	m_all_roi[faceth] = 3;
    // Calculate duration of template matching
    m_templateMatchingCurrentTime[faceth] = cv::getTickCount();
    double duration = (double)(m_templateMatchingCurrentTime[faceth] - m_templateMatchingStartTime[faceth]) / TICK_FREQUENCY;

    // If template matching lasts for more than 2 seconds face is possibly lost
    // so disable it and redetect using cascades
    if (duration > m_templateMatchingMaxDuration) {
        m_foundFace = false;
        m_templateMatchingRunning = false;
        m_templateMatchingStartTime[faceth] = m_templateMatchingCurrentTime[faceth] = 0;
		m_facePositions[faceth].x = m_facePositions[faceth].y = 0;
		m_trackedFaces[faceth].x = m_trackedFaces[faceth].y = m_trackedFaces[faceth].width = m_trackedFaces[faceth].height = 0;
		return;
    }

	// Edge case when face exits frame while 
	if (m_faceTemplates[faceth].rows * m_faceTemplates[faceth].cols == 0 || m_faceTemplates[faceth].rows <= 1 || m_faceTemplates[faceth].cols <= 1) {
		m_foundFace = false;
		m_templateMatchingRunning = false;
		m_templateMatchingStartTime[faceth] = m_templateMatchingCurrentTime[faceth] = 0;
		m_facePositions[faceth].x = m_facePositions[faceth].y = 0;
		m_trackedFaces[faceth].x = m_trackedFaces[faceth].y = m_trackedFaces[faceth].width = m_trackedFaces[faceth].height = 0;
		return;
	}

    // Template matching with last known face 
    //cv::matchTemplate(frame(m_faceRoi), m_faceTemplate, m_matchingResult, CV_TM_CCOEFF);
	cv::matchTemplate(frame(m_faceRois[faceth]), m_faceTemplates[faceth], m_matchingResult, CV_TM_SQDIFF_NORMED);
    cv::normalize(m_matchingResult, m_matchingResult, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
    double min, max;
    cv::Point minLoc, maxLoc;

    cv::minMaxLoc(m_matchingResult, &min, &max, &minLoc, &maxLoc);

    // Add roi offset to face position
	minLoc.x += m_faceRois[faceth].x;
	minLoc.y += m_faceRois[faceth].y;

    // Get detected face
	m_trackedFaces[faceth] = cv::Rect(minLoc.x, minLoc.y, m_faceTemplates[faceth].cols, m_faceTemplates[faceth].rows);
	m_trackedFaces[faceth] = doubleRectSize(m_trackedFaces[faceth], cv::Rect(0, 0, frame.cols, frame.rows));

    // Get new face template
	m_faceTemplates[faceth] = getFaceTemplate(frame, m_trackedFaces[faceth]);

    // Calculate face roi
	m_faceRois[faceth] = doubleRectSize(m_trackedFaces[faceth], cv::Rect(0, 0, frame.cols, frame.rows));

    // Update face position
	m_facePositions[faceth] = centerOfRect(m_trackedFaces[faceth]);
}

void VideoFaceDetector::Thread_All_Search(cv::Mat resizedFrame, int &result){
	std::vector<cv::Rect> all_Faces;
	m_faceCascadethread->detectMultiScale(resizedFrame, all_Faces, 1.1, 3, 0,
		cv::Size(resizedFrame.rows / 5, resizedFrame.rows / 5),
		cv::Size(resizedFrame.rows * 2 / 3, resizedFrame.rows * 2 / 3));
	result = all_Faces.size();
}

void VideoFaceDetector::Delete_Contain(std::vector<cv::Rect>&Input_Rects, std::vector<cv::Point>&Input_Points, int &facenum)
{
	for (int i =0; i < facenum; i++)
	{
		for (int j = i + 1; j < facenum; j++)
		{
			if ((Input_Rects[j].x >= Input_Rects[i].x && Input_Rects[j].y >= Input_Rects[i].y && \
				(Input_Rects[j].x + Input_Rects[j].width) <= (Input_Rects[i].x + Input_Rects[i].width) && \
				(Input_Rects[j].y + Input_Rects[j].height) <= (Input_Rects[i].y + Input_Rects[i].height)))
			{
				--facenum;
				Input_Rects[j] = Input_Rects[facenum];
				Input_Points[j] = Input_Points[facenum];
			}
		}
	}
}

std::vector<cv::Point> VideoFaceDetector::getFrameAndDetect(cv::Mat &frame)
{
    *m_videoCapture >> frame;

	if (!frame.empty()){
		// Downscale frame to m_resizedWidth width - keep aspect ratio
		m_scale = (double)std::min(m_resizedWidth, frame.cols) / frame.cols;
		cv::Size resizedFrameSize = cv::Size((int)(m_scale * frame.cols), (int)(m_scale * frame.rows));

		cv::Mat resizedFrame;
		cv::resize(frame, resizedFrame, resizedFrameSize);

		m_allsearchFaceCurrentTime = cv::getTickCount();
		double duration = (double)(m_allsearchFaceCurrentTime - m_allsearchFaceStartTime) / TICK_FREQUENCY;

		std::thread first(&VideoFaceDetector::Thread_All_Search, this, resizedFrame, std::ref(m_jump_allSearch));

		//当全局扫描时间超过3秒的时候重新进行全局扫描
		if (!m_foundFace || (m_jump_allSearch - m_face_num) > 0 || duration > m_allsearchFaceMaxDuration){
			detectFaceAllSizes(resizedFrame); // Detect using cascades over whole image
			m_allsearchFaceStartTime = cv::getTickCount();
		}
		else {
			for (int i = 0; i < m_face_num; i++){
				detectFaceAroundRoi(resizedFrame, i); // Detect using cascades only in ROI
				if (m_templateMatchingRunning) {
					detectFacesTemplateMatching(resizedFrame, i); // Detect using template matching
				}
			}
			Delete_Contain(m_roitrackedFace, m_facePositions, m_face_num);
		}
		first.join();
	}
    return m_facePositions;
}

std::vector<cv::Point> VideoFaceDetector::operator>>(cv::Mat &frame)
{
    return this->getFrameAndDetect(frame);
}