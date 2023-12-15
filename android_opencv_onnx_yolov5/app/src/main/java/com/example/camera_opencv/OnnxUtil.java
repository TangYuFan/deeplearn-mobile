package com.example.camera_opencv;

import ai.onnxruntime.*;
import android.content.res.AssetManager;
import com.alibaba.fastjson.JSONObject;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.*;

public class OnnxUtil {

    // onnxruntime 环境
    public static OrtEnvironment env;
    public static OrtSession session;

    // 模型输入
    public static int w = 0;
    public static int h = 0;
    public static int c = 3;

    // 类别信息
    static List<String> clazzStr;

    // 标注颜色
    public static Scalar color = new Scalar(255, 0, 255);
    public static int tickness = 1;

    // 模型加载
    public static void loadModule(AssetManager assetManager){

        // 模型最大输入是 640*640,为了提高帧率这里选择 160*160
        w = 256;h = 256;c = 3;
//        w = 128;h = 128;c = 3;
//        w = 640;h = 640;c = 3;

        try {

//            InputStream inputStream = assetManager.open("yolov5n_640x640.onnx");
            InputStream inputStream = assetManager.open("yolov5n_256x256.onnx");
//            InputStream inputStream = assetManager.open("yolov5n_128x128.onnx");
            ByteArrayOutputStream buffer = new ByteArrayOutputStream();
            int nRead;
            byte[] data = new byte[1024];
            while ((nRead = inputStream.read(data, 0, data.length)) != -1) {
                buffer.write(data, 0, nRead);
            }
            buffer.flush();
            byte[] module = buffer.toByteArray();
            System.out.println("开始加载模型");
            env = OrtEnvironment.getEnvironment();
            session = env.createSession(module, new OrtSession.SessionOptions());
            session.getInputInfo().entrySet().stream().forEach(n -> {
                String inputName = n.getKey();
                NodeInfo inputInfo = n.getValue();
                long[] shape = ((TensorInfo) inputInfo.getInfo()).getShape();
                String javaType = ((TensorInfo) inputInfo.getInfo()).type.toString();
                System.out.println("模型输入:  "+inputName + " -> " + Arrays.toString(shape) + " -> " + javaType);
            });
            session.getOutputInfo().entrySet().stream().forEach(n -> {
                String outputName = n.getKey();
                NodeInfo outputInfo = n.getValue();
                long[] shape = ((TensorInfo) outputInfo.getInfo()).getShape();
                String javaType = ((TensorInfo) outputInfo.getInfo()).type.toString();
                System.out.println("模型输出:  "+outputName + " -> " + Arrays.toString(shape) + " -> " + javaType);
            });
            // 类别信息
            JSONObject clazz = JSONObject.parseObject(session.getMetadata().getCustomMetadata().get("names").replace("\"","\"\""));
            clazzStr = new ArrayList<>();
            clazz.entrySet().forEach(n->{
                clazzStr.add(String.valueOf(n.getValue()));
            });

        } catch (Exception e) {
            e.printStackTrace();
        }

    }


    // 模型推理,输入原始图片和识别区域两个点
    public static void inference1(Mat mat, Point detection_p1, Point detection_p2){

        int px = Double.valueOf(detection_p1.x).intValue();
        int py = Double.valueOf(detection_p1.y).intValue();

        // 提取rgb,java预处理
//        float[][][][] chw = new float[1][c][h][w];
//        for(int j=0 ; j<h ; j++){
//            for(int i=0 ; i<w ; i++){
//                // 第j行,第i列,根据识别区域p1得到xy坐标的偏移,直接加就行
//                double[] rgb = mat.get(j+py,i+px);
//                // 缓存到 chw 中,mat 是 rgba 数据对应的下标 2103
//                chw[0][0][j][i] = (float)(rgb[2]/255);//r
//                chw[0][1][j][i] = (float)(rgb[1]/255);//G
//                chw[0][2][j][i] = (float)(rgb[0]/255);//b
//            }
//        }

        long t1 = System.currentTimeMillis();

        // opencv 预处理,先截取识区域,xywh
        Mat sub = new Mat(mat, new Rect(
                Double.valueOf(detection_p1.x).intValue(),
                Double.valueOf(detection_p1.y).intValue(),
                Double.valueOf(detection_p2.x-detection_p1.x).intValue(),
                Double.valueOf(detection_p2.y-detection_p1.y).intValue())).clone();
        // 转RGB并归一化,提取到数组中,hwc转chw
        Imgproc.cvtColor(sub, sub, Imgproc.COLOR_BGR2RGB);
        sub.convertTo(sub, CvType.CV_32FC1, 1. / 255);
        // 提取到数组中 hwc
        float[] hwc = new float[ c*h*w];
        sub.get(0,0,hwc);
        float[] chw = hwc2chw(hwc);


        long t2 = System.currentTimeMillis();
        System.out.println("预处理耗时:"+(t2-t1)+"毫秒");

        try {

            float confThreshold = 0.35f;
            float nmsThreshold = 0.45f;
//            OnnxTensor tensor = OnnxTensor.createTensor(env, chw);
            OnnxTensor tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(chw), new long[]{1,c,h,w});
            OrtSession.Result out = session.run(Collections.singletonMap("images", tensor));
            float[][] data = ((float[][][])(out.get(0)).getValue())[0];

            List<Detection> box_before_nsm = new ArrayList<>();
            List<Detection> box_after_nsm = new ArrayList<>();
            for(int i=0;i<data.length;i++){
                float[] obj = data[i];
                if(obj[4]>=confThreshold){
                    box_before_nsm.add(new Detection(obj));
                }
            }

            box_before_nsm.sort((o1, o2) -> Float.compare(o2.type_max_value,o1.type_max_value));
            while (!box_before_nsm.isEmpty()){
                Detection maxObj = box_before_nsm.get(0);
                box_after_nsm.add(maxObj);
                Iterator<Detection> it = box_before_nsm.iterator();
                while (it.hasNext()) {
                    Detection obj = it.next();
                    // 计算交并比
                    if(Detection.calculateIoU(maxObj,obj)>nmsThreshold){
                        it.remove();
                    }
                }
            }

            // 标注
            box_after_nsm.stream().forEach(n->{
                // 边框
                Imgproc.rectangle(
                        mat,
                        new Point(n.x1+px,n.y1+py),
                        new Point(n.x2+px,n.y2+py),
                        color,
                        tickness
                );
                // 类别
                Imgproc.putText(
                        mat,
                        n.type_max_name+" "+String.format("%.2f", n.type_max_value*100)+"%",
                        new Point(n.x1+px,n.y1+py-1),
                        1,
                        tickness,
                        color
                );
            });
            tensor.close();
            out.close();
        }
        catch (Exception e){
            e.printStackTrace();
        }

    }

    public static void inference2(Mat mat,Point detection_p1,Point detection_p2){

        int px = Double.valueOf(detection_p1.x).intValue();
        int py = Double.valueOf(detection_p1.y).intValue();


        long t1 = System.currentTimeMillis();

        // opencv 预处理,先截取识区域,xywh
        Mat sub = new Mat(mat, new Rect(
                Double.valueOf(detection_p1.x).intValue(),
                Double.valueOf(detection_p1.y).intValue(),
                Double.valueOf(detection_p2.x-detection_p1.x).intValue(),
                Double.valueOf(detection_p2.y-detection_p1.y).intValue()));
        // 转RGB并归一化,提取到数组中,hwc转chw
        Imgproc.cvtColor(sub, sub, Imgproc.COLOR_BGR2RGB);
        sub.convertTo(sub, CvType.CV_32FC1, 1. / 255);
        // 提取rgb mat
        List<Mat> mats = new ArrayList<>();
        Core.split(sub, mats);
        Mat r = mats.get(0);
        Mat g = mats.get(0);
        Mat b = mats.get(0);
        float[] r_data = new float[r.cols()*r.rows()];
        float[] g_data = new float[g.cols()*g.rows()];
        float[] b_data = new float[b.cols()*b.rows()];
        r.get(0, 0, r_data);
        g.get(0, 0, g_data);
        b.get(0, 0, b_data);

        // 因为输入是chw 也就是说  rrrrr ggggg bbbbb 将三个数组拼接即可
        float[] chw = new float[ r_data.length + g_data.length + b_data.length ];
        int currentIndex = 0;
        System.arraycopy(r_data, 0, chw, currentIndex, r_data.length);currentIndex += r_data.length;
        System.arraycopy(g_data, 0, chw, currentIndex, g_data.length);currentIndex += g_data.length;
        System.arraycopy(b_data, 0, chw, currentIndex, b_data.length);

        sub.release();

        long t2 = System.currentTimeMillis();
        System.out.println("预处理耗时:"+(t2-t1)+"毫秒");

        try {

            long t3 = System.currentTimeMillis();

            float confThreshold = 0.35f;
            float nmsThreshold = 0.45f;
            OnnxTensor tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(chw), new long[]{1,c,h,w});
            OrtSession.Result out = session.run(Collections.singletonMap("images", tensor));
            float[][] data = ((float[][][])(out.get(0)).getValue())[0];

            long t4 = System.currentTimeMillis();
            System.out.println("推理耗时:"+(t4-t3)+"毫秒");

            List<Detection> box_before_nsm = new ArrayList<>();
            List<Detection> box_after_nsm = new ArrayList<>();

            long t5 = System.currentTimeMillis();

            for(int i=0;i<data.length;i++){
                float[] obj = data[i];
                if(obj[4]>=confThreshold){
                    box_before_nsm.add(new Detection(obj));
                }
            }

            box_before_nsm.sort((o1, o2) -> Float.compare(o2.type_max_value,o1.type_max_value));
            while (!box_before_nsm.isEmpty()){
                Detection maxObj = box_before_nsm.get(0);
                box_after_nsm.add(maxObj);
                Iterator<Detection> it = box_before_nsm.iterator();
                while (it.hasNext()) {
                    Detection obj = it.next();
                    // 计算交并比
                    if(Detection.calculateIoU(maxObj,obj)>nmsThreshold){
                        it.remove();
                    }
                }
            }


            long t6 = System.currentTimeMillis();
            System.out.println("后处理耗时:"+(t6-t5)+"毫秒");


            // 标注
            box_after_nsm.stream().forEach(n->{
                // 边框
                Imgproc.rectangle(
                        mat,
                        new Point(n.x1+px,n.y1+py),
                        new Point(n.x2+px,n.y2+py),
                        color,
                        tickness
                );
                // 类别
                Imgproc.putText(
                        mat,
                        n.type_max_name+" "+String.format("%.2f", n.type_max_value*100)+"%",
                        new Point(n.x1+px,n.y1+py-1),
                        1,
                        tickness,
                        color
                );
            });

            long t7 = System.currentTimeMillis();
            System.out.println("标注耗时:"+(t7-t6)+"毫秒");


            tensor.close();
            out.close();
        }
        catch (Exception e){
            e.printStackTrace();
        }

    }
    // 目标框
    public static class  Detection{
        float x1;
        float y1;
        float x2;
        float y2;
        int type_max_index;
        float type_max_value;
        String type_max_name;
        public Detection(float[] box){
            // xywh
            float x = box[0];
            float y = box[1];
            float w = box[2];
            float h = box[3];
            // x1y1x2y2
            this.x1 = x - w * 0.5f;
            this.y1 = y - h * 0.5f;
            this.x2 = x + w * 0.5f;
            this.y2 = y + h * 0.5f;
            // 计算概率最大值index,第5位后面开始就是概率
            int max_index = 0;
            float max_value = 0;
            for (int i = 5; i < box.length; i++) {
                if (box[i] > max_value) {
                    max_value = box[i];
                    max_index = i;
                }
            }
            type_max_index = max_index - 5;
            type_max_value = max_value;
            type_max_name = clazzStr.get(type_max_index);
        }
        // 计算两个交并比
        private static double calculateIoU(Detection box1, Detection box2) {
            double x1 = Math.max(box1.x1, box2.x1);
            double y1 = Math.max(box1.y1, box2.y1);
            double x2 = Math.min(box1.x2, box2.x2);
            double y2 = Math.min(box1.y2, box2.y2);
            double intersectionArea = Math.max(0, x2 - x1 + 1) * Math.max(0, y2 - y1 + 1);
            double box1Area = (box1.x2 - box1.x1 + 1) * (box1.y2 - box1.y1 + 1);
            double box2Area = (box2.x2 - box2.x1 + 1) * (box2.y2 - box2.y1 + 1);
            double unionArea = box1Area + box2Area - intersectionArea;
            return intersectionArea / unionArea;
        }
    }


    public static float[] hwc2chw(float[] src) {
        float[] chw = new float[src.length];
        int j = 0;
        for (int ch = 0; ch < 3; ++ch) {
            for (int i = ch; i < src.length; i += 3) {
                chw[j] = src[i];
                j++;
            }
        }
        return chw;
    }
}




