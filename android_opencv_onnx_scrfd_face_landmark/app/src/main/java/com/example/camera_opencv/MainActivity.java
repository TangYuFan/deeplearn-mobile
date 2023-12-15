package com.example.camera_opencv;


import android.content.pm.ActivityInfo;
import android.os.Bundle;
import android.view.WindowManager;
import com.example.camera_opencv.databinding.ActivityMainBinding;
import org.opencv.android.*;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.util.Arrays;
import java.util.List;

public class MainActivity extends CameraActivity{

    // 动态库
    static {
        // 我们自己的jni
        System.loadLibrary("camera_opencv");
        // 新加的 opencv 的jni
        System.loadLibrary("opencv_java4");
    }

    private ActivityMainBinding binding;

    // 预览界面
    private JavaCamera2View camera2View;

    // 相机编号 0后置 1前置
    private int cameraId = 1;

    // 设置预览界面宽高,在次宽高基础上限制识别区域
    private int win_w = 320;
    private int win_h = 240;

    // 识别区域两个点
    private Point detection_p1;
    private Point detection_p2;
    private Scalar detection_box_color;
    private int detection_box_tickness;

    @Override
    protected void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);
        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        // 加载模型
        OnnxUtil.loadModule(getAssets());

        // 强制横屏
        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);
        // 隐藏上方状态栏
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN, WindowManager.LayoutParams.FLAG_FULLSCREEN);
        // 预览界面
        camera2View = findViewById(R.id.camera_view);
    }

    @Override
    protected List<? extends CameraBridgeViewBase> getCameraViewList() {
        return Arrays.asList(camera2View);
    }


    @Override
    public void onPause() {
        super.onPause();
        if (camera2View != null) {
            // 关闭预览
            camera2View.disableView();
        }
    }

    @Override
    public void onResume() {
        super.onResume();
        if (OpenCVLoader.initDebug()) {
            baseLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        } else {
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, baseLoaderCallback);
        }
    }

    // 获取每一帧回调数据
    private CameraBridgeViewBase.CvCameraViewListener2 cameraViewListener2 = new CameraBridgeViewBase.CvCameraViewListener2() {
        @Override
        public void onCameraViewStarted(int width, int height) {
            System.out.println("开始预览 width="+width+",height="+height);
            // 预览界面长宽分别是识别区域的两倍,识别区域正中间正常形区域
            int detection_x1 = (win_w - OnnxUtil.w)/2;
            int detection_x2 = (win_w - OnnxUtil.w)/2 + OnnxUtil.w;
            int detection_y1 = (win_h - OnnxUtil.h)/2;
            int detection_y2 = (win_h - OnnxUtil.h)/2 + OnnxUtil.h;;
            // 缓存识别区域两个点
            detection_p1 = new Point(detection_x1,detection_y1);
            detection_p2 = new Point(detection_x2,detection_y2);
            detection_box_color = new Scalar(255, 0, 0);
            detection_box_tickness = 1;
        }
        @Override
        public void onCameraViewStopped() {}
        @Override
        public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame frame) {

            // 获取 cv::Mat
            Mat mat = frame.rgba();

            // 标注识别区域
            Imgproc.rectangle(mat, detection_p1, detection_p2,detection_box_color,detection_box_tickness);
            Imgproc.putText(mat,"RecArea",detection_p1,Imgproc.FONT_HERSHEY_SIMPLEX,0.5,detection_box_color);

            // 推理并标注
//            OnnxUtil.inference1(mat,detection_p1,detection_p2); // java 预处理耗时50ms 10fps
            OnnxUtil.inference2(mat,detection_p1,detection_p2); // opencv 预处理耗时1ms 16fps
//            OnnxUtil.inference3(mat,detection_p1,detection_p2); // opencv 预处理耗时1ms 16fps

            return mat;
        }
    };

    // 开启预览
    private BaseLoaderCallback baseLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    if (camera2View != null) {
                        // 设置前置还是后置摄像头 0后置 1前置
                        camera2View.setCameraIndex(cameraId);
                        // 注册每一帧回调
                        camera2View.setCvCameraViewListener(cameraViewListener2);
                        // 显示/关闭 帧率  disableFpsMeter/enableFpsMeter
                        // 要修改字体和颜色直接修改 FpsMeter 类即可
                        camera2View.enableFpsMeter();
                        // 设置视图宽高和模型一致减少resize操作,模型输入一般尺寸不大,这样相机渲染fps会更高
                        camera2View.setMaxFrameSize(win_w,win_h);
                        // 开启
                        camera2View.enableView();
                    }
                }
                break;
                default:
                    super.onManagerConnected(status);
                    break;
            }
        }
    };

}