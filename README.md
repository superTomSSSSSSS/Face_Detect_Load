# Face_Detect_Load

  * 环境：VS2015 + opencv2.4.9
  * 检测到的正面人脸图片下载在Face_Image文件夹
  * 中心思想：首先全图扫描，扫描到人脸，定位ROI感兴趣区域，下一帧在感兴趣区域扫描，当感兴趣区域扫描不到人脸时，用上一帧检测到的人脸作为模板，去进行模板匹配，当模板匹配时间超过3秒认为跟踪人脸
消失，重新全图扫描。（人脸检测：Harr+级联，采用opencv内部训练好的特征集）  
    正脸图片即是全图扫描时，检测到的人脸矩形框。

```
//重载了>>，作为函数入口
std::vector<cv::Point> VideoFaceDetector::operator>>(cv::Mat &frame)
{
    return this->getFrameAndDetect(frame);
}
```

```
//当遇到三种情况进入detectFaceAllSizes(),进行全图扫描（1.没有搜索到人脸 2.人脸数目发生变化 3.扫描时间大于3s）
 if (!m_foundFace || (m_jump_allSearch - m_face_num) > 0 || duration > m_allsearchFaceMaxDuration){
	detectFaceAllSizes(resizedFrame);
 }
```
```
//m_jump_allSearch:(另开线程)时刻统计全图人脸的数目，与跟踪的m_face_num对比，如果有变化立即跳入全局扫描
 std::thread first(&VideoFaceDetector::Thread_All_Search, this, resizedFrame, std::ref(m_jump_allSearch));
```
```
 detectFaceAroundRoi(resizedFrame, i); //ROI人脸检测
 detectFacesTemplateMatching(resizedFrame, i); //模板匹配行人跟踪
```

```
//对检测到的矩形框做调整，除去包围矩阵
Delete_Contain(m_roitrackedFace, m_facePositions, m_face_num);
```

```
//主函数，判断m_all_roi[i] == 1时（即全图搜索时），load人脸图片
vector<int> m_all_roi; //定位跳入全局、局部还是模板匹配
```
