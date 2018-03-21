#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>

#include "VideoFaceDetector.h"

const cv::String    WINDOW_NAME("Camera video");
const cv::String    CASCADE_FILE("haarcascade_frontalface_default.xml");
const cv::String    CASCADE_COPY_FILE("haarcascade_frontalface_default_copy.xml");

int main(int argc, char** argv)
{
	// Try opening camera
	cv::VideoCapture camera(0);
	//cv::VideoCapture camera("D:\\video.mp4");
	if (!camera.isOpened()) {
		fprintf(stderr, "Error getting camera...\n");
		exit(1);
	}

	cv::namedWindow(WINDOW_NAME, cv::WINDOW_NORMAL | cv::WINDOW_AUTOSIZE);

	VideoFaceDetector detector(CASCADE_FILE, CASCADE_COPY_FILE, camera);
	cv::Mat frame;
	double fps = 0, time_per_frame;

	char image[20];  //Save Image
	int count_face = 0;

	while (true)
	{
		auto start = cv::getCPUTickCount();
		detector >> frame;
		auto end = cv::getCPUTickCount();

		time_per_frame = (end - start) / cv::getTickFrequency();
		fps = (15 * fps + (1 / time_per_frame)) / 16;

		printf("Time per frame: %3.3f\tFPS: %3.3f  %3.3d  %3.3d\n", time_per_frame, fps, detector.m_face_num, detector.m_jump_allSearch);

		if (!frame.empty()){
			if (detector.isFaceFound())
			{
				for (int i = 0; i < detector.m_face_num; i++){

					if (detector.m_all_roi[i] == 1)
					{
						sprintf(image, "%s%d%s", "..\\Face_Image\\Face", ++count_face, ".jpg");   //The path of face image
						cv::Mat face = frame(detector.face(i));
						cv::imwrite(image, face);
					}
				}
			}

			cv::imshow(WINDOW_NAME, frame);
			if (cv::waitKey(25) == 27) break;
		}
	}

	return 0;
}