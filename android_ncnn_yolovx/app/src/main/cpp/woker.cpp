#include <android/asset_manager_jni.h>
#include <android/native_window_jni.h>
#include <android/native_window.h>

#include <android/log.h>
#include <jni.h>
#include <string>
#include <vector>
#include <platform.h>
#include <benchmark.h>
#include "ndkcamera.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "yolox.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

static int draw_unsupported(cv::Mat& rgb)
{
    const char text[] = "unopen rec det";
    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 1.0, 1, &baseLine);
    int y = (rgb.rows - label_size.height) / 2;
    int x = (rgb.cols - label_size.width) / 2;
    cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),cv::Scalar(255, 255, 255), -1);
    cv::putText(rgb, text, cv::Point(x, y + label_size.height),cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0));

    return 0;
}

// fps 和推理耗时显示
static int draw_fps(cv::Mat& rgb, double t1,int count)
{
    double t = t1;
    // resolve moving average
    float avg_fps = 0.f;
    {
        static double t0 = 0.f;
        static float fps_history[10] = {0.f};

        double t1 = ncnn::get_current_time();
        if (t0 == 0.f)
        {
            t0 = t1;
            return 0;
        }

        float fps = 1000.f / (t1 - t0);
        t0 = t1;

        for (int i = 9; i >= 1; i--)
        {
            fps_history[i] = fps_history[i - 1];
        }
        fps_history[0] = fps;

        if (fps_history[9] == 0.f)
        {
            return 0;
        }

        for (int i = 0; i < 10; i++)
        {
            avg_fps += fps_history[i];
        }
        avg_fps /= 10.f;
    }

    char text[32];
    sprintf(text, "%.2ffps %.2fms obj=%d",avg_fps,t,count);

    int baseLine = 0;

    // 计算文本字符串宽度和高度
    float fontScale = 0.5;
    int tickness = 1;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, fontScale, tickness, &baseLine);

    // 设置到屏幕左上角
    int y = 0;
    int x = 0;

    // 设置到屏幕右上脚
//       int y = 0;
//    int x = rgb.cols - label_size.width;

    // 文字区域设置背景
    cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),cv::Scalar(0, 0, 0), -1);

    // 设置文字
    cv::putText(rgb, text, cv::Point(x, y + label_size.height),cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,2555));

    return 0;
}

static ncnn::Mutex lock;
static Yolox* g_yolovx = 0;

class MyNdkCamera : public NdkCameraWindow{
public:
    virtual void on_image_render(cv::Mat& rgb) const;
};

void MyNdkCamera::on_image_render(cv::Mat& rgb) const{

    // 推理耗时
    double t;
    int count;
    // 在这里处理图片识别
    {
        // ncnn 的等待超时机制
        ncnn::MutexLockGuard g(lock);

        // 模型已初始化
        if(g_yolovx){
            // 推理,输入两个阈值返回推理耗时
            std::vector<Object> objects;
            t = g_yolovx->detect(rgb, objects,0.45f,0.65f);
            // 标注
            g_yolovx->draw(rgb, objects);
            // 目标个数
            count = objects.size();
        }
        // 绘制提示信息
        else{
            draw_unsupported(rgb);
        }
    }

    // 显示fps 推理耗时 目标个数
    draw_fps(rgb, t,count);
}



//----------------------------------------------下面是所有jni定义------------------------------------------------------

static MyNdkCamera* g_camera = 0;

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "JNI_OnLoad");

    g_camera = new MyNdkCamera;

    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "JNI_OnUnload");
    {
        ncnn::MutexLockGuard g(lock);
    }
    delete g_camera;
    g_camera = 0;
}


// 打开相机
JNIEXPORT jboolean JNICALL Java_camera_demo_Woker_openCamera(JNIEnv* env, jobject thiz, jint facing)
{
    if (facing < 0 || facing > 1)
        return JNI_FALSE;
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "openCamera %d", facing);
    g_camera->open((int)facing);
    return JNI_TRUE;
}

// 关闭相机
JNIEXPORT jboolean JNICALL Java_camera_demo_Woker_closeCamera(JNIEnv* env, jobject thiz)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "closeCamera");
    g_camera->close();
    return JNI_TRUE;
}

// 帧输出
JNIEXPORT jboolean JNICALL Java_camera_demo_Woker_setOutputWindow(JNIEnv* env, jobject thiz, jobject surface)
{
    ANativeWindow* win = ANativeWindow_fromSurface(env, surface);
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "setOutputWindow %p", win);
    g_camera->set_window(win);
    return JNI_TRUE;
}

// 模型初始化
JNIEXPORT jboolean JNICALL Java_camera_demo_Woker_loadModule(JNIEnv* env, jobject thiz, jobject assetManager)
{

    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "loadModel mgr=%p", mgr);

    // 初始化
    ncnn::MutexLockGuard g(lock);

    // 初始化 yolox对象
    if (!g_yolovx){
        g_yolovx = new Yolox;
    }
    // 加载模型
    const char* modeltype = "yolox-nano"; // yolox-nano 或 yolox-tiny
    const int target_sizes = 416;
    const float mean_vals[] = {255.f * 0.485f, 255.f * 0.456, 255.f * 0.406f};
    const float norm_vals[] = {1 / (255.f * 0.229f), 1 / (255.f * 0.224f), 1 / (255.f * 0.225f)};
    g_yolovx->load(mgr,modeltype,target_sizes,mean_vals,norm_vals);


    return JNI_TRUE;
}

}
