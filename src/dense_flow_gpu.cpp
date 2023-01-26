#include "common.h"
#include "dense_flow.h"

#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudaoptflow.hpp"
#include "opencv2/cudacodec.hpp"

#include <stdio.h>
#include <iostream>

#include "warp_flow.h"
#include "path_tools.h"

using namespace cv;
using namespace cv::cuda;
using namespace std;

void calcDenseFlowGPU(string file_name, string dump_path, int bound, int type, int step, int dev_id){
    VideoCapture video_stream(file_name);
    CHECK(video_stream.isOpened())<<"Cannot open video stream \""
                                  <<file_name
                                  <<"\" for optical flow extraction.";
	
    auto avg_fps = video_stream.get(cv::CAP_PROP_FPS);
	auto width = video_stream.get(cv::CAP_PROP_FRAME_WIDTH);
	auto height = video_stream.get(cv::CAP_PROP_FRAME_HEIGHT);

	cv::VideoWriter wrt_x(safelyJoinPath(dump_path, "x.mp4"), cv::VideoWriter::fourcc('m','p','4','v'), 
                            avg_fps, cv::Size(int(width), int(height)), 0);
	cv::VideoWriter wrt_y(safelyJoinPath(dump_path, "y.mp4"), cv::VideoWriter::fourcc('m','p','4','v'), 
						avg_fps, cv::Size(int(width), int(height)), 0);

    setDevice(dev_id);
    Mat capture_frame, capture_image, prev_image, capture_gray, prev_gray;
    Mat flow_x, flow_y;

    GpuMat d_frame_0, d_frame_1;
    GpuMat d_flow;

    cv::Ptr<cuda::FarnebackOpticalFlow> alg_farn = cuda::FarnebackOpticalFlow::create();
    cv::Ptr<cuda::OpticalFlowDual_TVL1> alg_tvl1 = cuda::OpticalFlowDual_TVL1::create();
    cv::Ptr<cuda::BroxOpticalFlow> alg_brox      = cuda::BroxOpticalFlow::create(0.197f, 50.0f, 0.8f, 10, 77, 10);

    bool initialized = false;
    int cnt = 0;
    while(true){

        //build mats for the first frame
        if (!initialized){
            video_stream >> capture_frame;
            if (capture_frame.empty()) break; // read frames until end
            initializeMats(capture_frame, capture_image, capture_gray,
                            prev_image, prev_gray);
            capture_frame.copyTo(prev_image);
            cvtColor(prev_image, prev_gray, COLOR_BGR2GRAY);

            initialized = true;
            for(int s = 0; s < step; ++s){
                video_stream >> capture_frame;
                cnt ++;
                if (capture_frame.empty()) return; // read frames until end
            }
        }else {
            capture_frame.copyTo(capture_image);
            cvtColor(capture_image, capture_gray, COLOR_BGR2GRAY);
            d_frame_0.upload(prev_gray);
            d_frame_1.upload(capture_gray);

            switch(type){
                case 0: {
                    alg_farn->calc(d_frame_0, d_frame_1, d_flow);
                    break;
                }
                case 1: {
                    alg_tvl1->calc(d_frame_0, d_frame_1, d_flow);
                    break;
                }
                case 2: {
                    GpuMat d_buf_0, d_buf_1;
                    d_frame_0.convertTo(d_buf_0, CV_32F, 1.0 / 255.0);
                    d_frame_1.convertTo(d_buf_1, CV_32F, 1.0 / 255.0);
                    alg_brox->calc(d_buf_0, d_buf_1, d_flow);
                    break;
                }
                default:
                    LOG(ERROR)<<"Unknown optical method: "<<type;
            }

            GpuMat planes[2];
            cuda::split(d_flow, planes);

            //get back flow map
            Mat flow_x(planes[0]);
            Mat flow_y(planes[1]);

			Mat flow_img_x(flow_x.size(), CV_8UC1);
    		Mat flow_img_y(flow_y.size(), CV_8UC1);

            convertFlowToImage(flow_x, flow_y, flow_img_x, flow_img_y, -bound, bound);

			wrt_x << flow_img_x;
			wrt_y << flow_img_y;

            std::swap(prev_gray, capture_gray);
            std::swap(prev_image, capture_image);
            
            //prefetch while gpu is working
            bool hasnext = true;
            for(int s = 0; s < step; ++s){
                video_stream >> capture_frame;
		        cnt ++;
                hasnext = !capture_frame.empty();
                // read frames until end
            }
            
            if (!hasnext){
                return;
            }
        }


    }

}
