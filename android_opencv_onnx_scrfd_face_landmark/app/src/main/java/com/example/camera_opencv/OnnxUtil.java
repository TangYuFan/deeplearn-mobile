package com.example.camera_opencv;

import ai.onnxruntime.*;
import android.content.res.AssetManager;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.*;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class OnnxUtil {

    // onnxruntime 环境
    public static OrtEnvironment env;
    public static OrtSession session;

    // 模型输入
    public static int w = 0;
    public static int h = 0;
    public static int c = 3;

    // 标注颜色
    public static Scalar green = new Scalar(0, 255, 0);
    public static int tickness = 1;

    // 多线程预处理
    public static ExecutorService fixedThreadPool = Executors.newFixedThreadPool(5);


    // 模型加载
    public static void loadModule(AssetManager assetManager){

        // 模型最大输入是 640*640,为了提高帧率这里选择 160*160
        w = 160;
        h = 160;
        c = 3;

        try {

            InputStream inputStream = assetManager.open("scrfd_500m_bnkps.onnx");
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
        } catch (Exception e) {
            e.printStackTrace();
        }

    }


    // 模型推理,输入原始图片和识别区域两个点
    public static void inference1(Mat mat, Point detection_p1, Point detection_p2){

        int px = Double.valueOf(detection_p1.x).intValue();
        int py = Double.valueOf(detection_p1.y).intValue();

        long t1 = System.currentTimeMillis();

        // 提取rgb(chw存储)并做归一化,也就是 rrrrr bbbbb ggggg
        float[][][][] chw = new float[1][c][h][w];
        CountDownLatch lock = new CountDownLatch(4);
        fixedThreadPool.execute(()->{
            for(int j=0 ; j<h/2 ; j++){
                for(int i=0 ; i<w/2 ; i++){
                    // 第j行,第i列,根据识别区域p1得到xy坐标的偏移,直接加就行
                    double[] rgb = mat.get(j+py,i+px);
                    // 缓存到 chw 中,mat 是 rgba 数据对应的下标 2103
                    chw[0][0][j][i] = (float)(rgb[2]/255);//r
                    chw[0][1][j][i] = (float)(rgb[1]/255);//G
                    chw[0][2][j][i] = (float)(rgb[0]/255);//b
                }
            }
            lock.countDown();
        });
        fixedThreadPool.execute(()->{
            for(int j=h/2 ; j<h ; j++){
                for(int i=w/2 ; i<w ; i++){
                    // 第j行,第i列,根据识别区域p1得到xy坐标的偏移,直接加就行
                    double[] rgb = mat.get(j+py,i+px);
                    // 缓存到 chw 中,mat 是 rgba 数据对应的下标 2103
                    chw[0][0][j][i] = (float)(rgb[2]/255);//r
                    chw[0][1][j][i] = (float)(rgb[1]/255);//G
                    chw[0][2][j][i] = (float)(rgb[0]/255);//b
                }
            }
            lock.countDown();
        });

        fixedThreadPool.execute(()->{
            for(int j=0 ; j<h/2 ; j++){
                for(int i=w/2 ; i<w ; i++){
                    // 第j行,第i列,根据识别区域p1得到xy坐标的偏移,直接加就行
                    double[] rgb = mat.get(j+py,i+px);
                    // 缓存到 chw 中,mat 是 rgba 数据对应的下标 2103
                    chw[0][0][j][i] = (float)(rgb[2]/255);//r
                    chw[0][1][j][i] = (float)(rgb[1]/255);//G
                    chw[0][2][j][i] = (float)(rgb[0]/255);//b
                }
            }
            lock.countDown();
        });

        fixedThreadPool.execute(()->{
            for(int j=h/2 ; j<h ; j++){
                for(int i=0 ; i<w/2 ; i++){
                    // 第j行,第i列,根据识别区域p1得到xy坐标的偏移,直接加就行
                    double[] rgb = mat.get(j+py,i+px);
                    // 缓存到 chw 中,mat 是 rgba 数据对应的下标 2103
                    chw[0][0][j][i] = (float)(rgb[2]/255);//r
                    chw[0][1][j][i] = (float)(rgb[1]/255);//G
                    chw[0][2][j][i] = (float)(rgb[0]/255);//b
                }
            }
            lock.countDown();
        });

        try {
            lock.await();
        }
        catch (Exception e){
            System.exit(0);
        }

        long t2 = System.currentTimeMillis();
        System.out.println("预处理耗时:"+(t2-t1)+"毫秒");

        // 创建张量并进行推理
        try {

            // ---------模型[2]输入-----------
            // input.1 -> [1, 3, -1, -1] -> FLOAT
            // ---------模型[2]输出-----------
            // 447 -> [12800, 1] -> FLOAT
            // 473 -> [3200, 1] -> FLOAT
            // 499 -> [800, 1] -> FLOAT
            // 450 -> [12800, 4] -> FLOAT
            // 476 -> [3200, 4] -> FLOAT
            // 502 -> [800, 4] -> FLOAT
            // 453 -> [12800, 10] -> FLOAT
            // 479 -> [3200, 10] -> FLOAT
            // 505 -> [800, 10] -> FLOAT
            OnnxTensor tensor = OnnxTensor.createTensor(env, chw);
            OrtSession.Result out = session.run(Collections.singletonMap("input.1", tensor));

            // 打印输入张量shape
            long[] shape = tensor.getInfo().getShape();

            // 置信度阈值和iou阈值
            float score_thres = 0.5f;
            float iou_thres = 0.7f;

            // 因为这里没有进行缩放,直接将检测区域的 160*160 输入模型
            float imgScale = 1.0f;

            // 检测步长
            List<float[]> temp = new ArrayList<>();
            int[] net_stride = new int[]{8, 16, 32};
            for(int index = 0;index < net_stride.length;index++){
                int stride = net_stride[index];
                float[][] scores = (float[][]) out.get(index).getValue();
                float[][] boxes = (float[][]) out.get(index + 3).getValue();
                float[][] points = (float[][]) out.get(index + 6).getValue();
                int ws = (int) Math.ceil(1.0f * shape[3] / stride);
                // 人脸框的个数
                int count = scores.length;
                for(int i=0;i<count;i++){
                    float score = scores[i][0];// 分数
                    if(score >= score_thres){
                        int anchorIndex = i / 2;
                        int rowNum = anchorIndex / ws;
                        int colNum = anchorIndex % ws;
                        //计算人脸框,坐标缩放到原始图片中
                        float anchorX = colNum * net_stride[index];
                        float anchorY = rowNum * net_stride[index];
                        float x1 = (anchorX - boxes[i][0] * net_stride[index])  * imgScale;
                        float y1 = (anchorY - boxes[i][1] * net_stride[index])  * imgScale;
                        float x2 = (anchorX + boxes[i][2] * net_stride[index])  * imgScale;
                        float y2 = (anchorY + boxes[i][3] * net_stride[index])  * imgScale;
                        // 关键点集合
                        float [] point = points[i];
                        // 5个关键点,坐标缩放到原始图片中
                        float pointX_1 = (point[0] * net_stride[index] + anchorX)  * imgScale;
                        float pointY_1 = (point[1] * net_stride[index] + anchorY)  * imgScale;
                        float pointX_2 = (point[2] * net_stride[index] + anchorX)  * imgScale;
                        float pointY_2 = (point[3] * net_stride[index] + anchorY)  * imgScale;
                        float pointX_3 = (point[4] * net_stride[index] + anchorX)  * imgScale;
                        float pointY_3 = (point[5] * net_stride[index] + anchorY)  * imgScale;
                        float pointX_4 = (point[6] * net_stride[index] + anchorX)  * imgScale;
                        float pointY_4 = (point[7] * net_stride[index] + anchorY)  * imgScale;
                        float pointX_5 = (point[8] * net_stride[index] + anchorX)  * imgScale;
                        float pointY_5 = (point[8] * net_stride[index] + anchorY)  * imgScale;
                        // 保存到tmp中,注意x1y1x2y2不能超出 w*h
                        temp.add(new float[]{
                                score,
                                x1>w?w:x1,
                                y1>h?h:y1,
                                x2>w?w:x2,
                                y2>h?h:y2,
                                pointX_1,pointY_1,
                                pointX_2,pointY_2,
                                pointX_3,pointY_3,
                                pointX_4,pointY_4,
                                pointX_5,pointY_5
                        });
                    }
                }
            }

            // nms
            ArrayList<float[]> datas_after_nms = new ArrayList<>();
            while (!temp.isEmpty()){
                float[] max = temp.get(0);
                datas_after_nms.add(max);
                Iterator<float[]> it = temp.iterator();
                while (it.hasNext()) {
                    // nsm阈值
                    float[] obj = it.next();
                    // x1y1x2y2 是 1234
                    double iou = calculateIoU(max,obj);
                    if (iou > iou_thres) {
                        it.remove();
                    }
                }
            }

            // 标注
            datas_after_nms.stream().forEach(n->{

                // 画边框和关键点需要添加偏移
                int x1 = Float.valueOf(n[1]).intValue() + px;
                int y1 = Float.valueOf(n[2]).intValue() + py;
                int x2 = Float.valueOf(n[3]).intValue() + px;
                int y2 = Float.valueOf(n[4]).intValue() + py;
                Imgproc.rectangle(mat, new Point(x1, y1), new Point(x2, y2), green, tickness);

                float point1_x = Float.valueOf(n[5]).intValue() + px;// 关键点1
                float point1_y = Float.valueOf(n[6]).intValue() + py;//
                float point2_x = Float.valueOf(n[7]).intValue() + px;// 关键点2
                float point2_y = Float.valueOf(n[8]).intValue() + py;//
                float point3_x = Float.valueOf(n[9]).intValue() + px;// 关键点3
                float point3_y = Float.valueOf(n[10]).intValue() + py;//

                Imgproc.circle(mat, new Point(point1_x, point1_y), 1, green, tickness);
                Imgproc.circle(mat, new Point(point2_x, point2_y), 1, green, tickness);
                Imgproc.circle(mat, new Point(point3_x, point3_y), 1, green, tickness);

            });


            tensor.close();
            out.close();
        }
        catch (Exception e){
            e.printStackTrace();
        }
    }

    // 模型推理,输入原始图片和识别区域两个点
    public static void inference2(Mat mat, Point detection_p1, Point detection_p2){

        int px = Double.valueOf(detection_p1.x).intValue();
        int py = Double.valueOf(detection_p1.y).intValue();

        long t1 = System.currentTimeMillis();

        // 提取rgb(chw存储)并做归一化,也就是 rrrrr bbbbb ggggg
        // opencv 预处理,先截取识区域,xywh
        Mat sub = new Mat(mat, new Rect(
                Double.valueOf(detection_p1.x).intValue(),
                Double.valueOf(detection_p1.y).intValue(),
                Double.valueOf(detection_p2.x-detection_p1.x).intValue(),
                Double.valueOf(detection_p2.y-detection_p1.y).intValue()));
        // 转RGB并归一化,提取到数组中,hwc转chw
        Imgproc.cvtColor(sub, sub, Imgproc.COLOR_BGR2RGB);
        sub.convertTo(sub, CvType.CV_32FC1, 1. / 255);
        // 提取到数组中 hwc
        float[] hwc = new float[ c*h*w];
        sub.get(0,0,hwc);
        float[] chw = hwc2chw(hwc);


        long t2 = System.currentTimeMillis();
        System.out.println("预处理耗时:"+(t2-t1)+"毫秒");

        // 创建张量并进行推理
        try {

            // ---------模型[2]输入-----------
            // input.1 -> [1, 3, -1, -1] -> FLOAT
            // ---------模型[2]输出-----------
            // 447 -> [12800, 1] -> FLOAT
            // 473 -> [3200, 1] -> FLOAT
            // 499 -> [800, 1] -> FLOAT
            // 450 -> [12800, 4] -> FLOAT
            // 476 -> [3200, 4] -> FLOAT
            // 502 -> [800, 4] -> FLOAT
            // 453 -> [12800, 10] -> FLOAT
            // 479 -> [3200, 10] -> FLOAT
            // 505 -> [800, 10] -> FLOAT
            OnnxTensor tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(chw), new long[]{1,c,h,w});
            OrtSession.Result out = session.run(Collections.singletonMap("input.1", tensor));

            // 打印输入张量shape
            long[] shape = tensor.getInfo().getShape();

            // 置信度阈值和iou阈值
            float score_thres = 0.5f;
            float iou_thres = 0.7f;

            // 因为这里没有进行缩放,直接将检测区域的 160*160 输入模型
            float imgScale = 1.0f;

            // 检测步长
            List<float[]> temp = new ArrayList<>();
            int[] net_stride = new int[]{8, 16, 32};
            for(int index = 0;index < net_stride.length;index++){
                int stride = net_stride[index];
                float[][] scores = (float[][]) out.get(index).getValue();
                float[][] boxes = (float[][]) out.get(index + 3).getValue();
                float[][] points = (float[][]) out.get(index + 6).getValue();
                int ws = (int) Math.ceil(1.0f * shape[3] / stride);
                // 人脸框的个数
                int count = scores.length;
                for(int i=0;i<count;i++){
                    float score = scores[i][0];// 分数
                    if(score >= score_thres){
                        int anchorIndex = i / 2;
                        int rowNum = anchorIndex / ws;
                        int colNum = anchorIndex % ws;
                        //计算人脸框,坐标缩放到原始图片中
                        float anchorX = colNum * net_stride[index];
                        float anchorY = rowNum * net_stride[index];
                        float x1 = (anchorX - boxes[i][0] * net_stride[index])  * imgScale;
                        float y1 = (anchorY - boxes[i][1] * net_stride[index])  * imgScale;
                        float x2 = (anchorX + boxes[i][2] * net_stride[index])  * imgScale;
                        float y2 = (anchorY + boxes[i][3] * net_stride[index])  * imgScale;
                        // 关键点集合
                        float [] point = points[i];
                        // 5个关键点,坐标缩放到原始图片中
                        float pointX_1 = (point[0] * net_stride[index] + anchorX)  * imgScale;
                        float pointY_1 = (point[1] * net_stride[index] + anchorY)  * imgScale;
                        float pointX_2 = (point[2] * net_stride[index] + anchorX)  * imgScale;
                        float pointY_2 = (point[3] * net_stride[index] + anchorY)  * imgScale;
                        float pointX_3 = (point[4] * net_stride[index] + anchorX)  * imgScale;
                        float pointY_3 = (point[5] * net_stride[index] + anchorY)  * imgScale;
                        float pointX_4 = (point[6] * net_stride[index] + anchorX)  * imgScale;
                        float pointY_4 = (point[7] * net_stride[index] + anchorY)  * imgScale;
                        float pointX_5 = (point[8] * net_stride[index] + anchorX)  * imgScale;
                        float pointY_5 = (point[8] * net_stride[index] + anchorY)  * imgScale;
                        // 保存到tmp中,注意x1y1x2y2不能超出 w*h
                        temp.add(new float[]{
                                score,
                                x1>w?w:x1,
                                y1>h?h:y1,
                                x2>w?w:x2,
                                y2>h?h:y2,
                                pointX_1,pointY_1,
                                pointX_2,pointY_2,
                                pointX_3,pointY_3,
                                pointX_4,pointY_4,
                                pointX_5,pointY_5
                        });
                    }
                }
            }

            // nms
            ArrayList<float[]> datas_after_nms = new ArrayList<>();
            while (!temp.isEmpty()){
                float[] max = temp.get(0);
                datas_after_nms.add(max);
                Iterator<float[]> it = temp.iterator();
                while (it.hasNext()) {
                    // nsm阈值
                    float[] obj = it.next();
                    // x1y1x2y2 是 1234
                    double iou = calculateIoU(max,obj);
                    if (iou > iou_thres) {
                        it.remove();
                    }
                }
            }

            // 标注
            datas_after_nms.stream().forEach(n->{

                // 画边框和关键点需要添加偏移
                int x1 = Float.valueOf(n[1]).intValue() + px;
                int y1 = Float.valueOf(n[2]).intValue() + py;
                int x2 = Float.valueOf(n[3]).intValue() + px;
                int y2 = Float.valueOf(n[4]).intValue() + py;
                Imgproc.rectangle(mat, new Point(x1, y1), new Point(x2, y2), green, tickness);

                float point1_x = Float.valueOf(n[5]).intValue() + px;// 关键点1
                float point1_y = Float.valueOf(n[6]).intValue() + py;//
                float point2_x = Float.valueOf(n[7]).intValue() + px;// 关键点2
                float point2_y = Float.valueOf(n[8]).intValue() + py;//
                float point3_x = Float.valueOf(n[9]).intValue() + px;// 关键点3
                float point3_y = Float.valueOf(n[10]).intValue() + py;//

                Imgproc.circle(mat, new Point(point1_x, point1_y), 1, green, tickness);
                Imgproc.circle(mat, new Point(point2_x, point2_y), 1, green, tickness);
                Imgproc.circle(mat, new Point(point3_x, point3_y), 1, green, tickness);

            });


            tensor.close();
            out.close();
        }
        catch (Exception e){
            e.printStackTrace();
        }
    }


    public static void inference3(Mat mat, Point detection_p1, Point detection_p2){

        int px = Double.valueOf(detection_p1.x).intValue();
        int py = Double.valueOf(detection_p1.y).intValue();

        long t1 = System.currentTimeMillis();

        // 提取rgb(chw存储)并做归一化,也就是 rrrrr bbbbb ggggg
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

        long t2 = System.currentTimeMillis();
        System.out.println("预处理耗时:"+(t2-t1)+"毫秒");

        // 创建张量并进行推理
        try {

            // ---------模型[2]输入-----------
            // input.1 -> [1, 3, -1, -1] -> FLOAT
            // ---------模型[2]输出-----------
            // 447 -> [12800, 1] -> FLOAT
            // 473 -> [3200, 1] -> FLOAT
            // 499 -> [800, 1] -> FLOAT
            // 450 -> [12800, 4] -> FLOAT
            // 476 -> [3200, 4] -> FLOAT
            // 502 -> [800, 4] -> FLOAT
            // 453 -> [12800, 10] -> FLOAT
            // 479 -> [3200, 10] -> FLOAT
            // 505 -> [800, 10] -> FLOAT
            OnnxTensor tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(chw), new long[]{1,c,h,w});
            OrtSession.Result out = session.run(Collections.singletonMap("input.1", tensor));

            // 打印输入张量shape
            long[] shape = tensor.getInfo().getShape();

            // 置信度阈值和iou阈值
            float score_thres = 0.5f;
            float iou_thres = 0.7f;

            // 因为这里没有进行缩放,直接将检测区域的 160*160 输入模型
            float imgScale = 1.0f;

            // 检测步长
            List<float[]> temp = new ArrayList<>();
            int[] net_stride = new int[]{8, 16, 32};
            for(int index = 0;index < net_stride.length;index++){
                int stride = net_stride[index];
                float[][] scores = (float[][]) out.get(index).getValue();
                float[][] boxes = (float[][]) out.get(index + 3).getValue();
                float[][] points = (float[][]) out.get(index + 6).getValue();
                int ws = (int) Math.ceil(1.0f * shape[3] / stride);
                // 人脸框的个数
                int count = scores.length;
                for(int i=0;i<count;i++){
                    float score = scores[i][0];// 分数
                    if(score >= score_thres){
                        int anchorIndex = i / 2;
                        int rowNum = anchorIndex / ws;
                        int colNum = anchorIndex % ws;
                        //计算人脸框,坐标缩放到原始图片中
                        float anchorX = colNum * net_stride[index];
                        float anchorY = rowNum * net_stride[index];
                        float x1 = (anchorX - boxes[i][0] * net_stride[index])  * imgScale;
                        float y1 = (anchorY - boxes[i][1] * net_stride[index])  * imgScale;
                        float x2 = (anchorX + boxes[i][2] * net_stride[index])  * imgScale;
                        float y2 = (anchorY + boxes[i][3] * net_stride[index])  * imgScale;
                        // 关键点集合
                        float [] point = points[i];
                        // 5个关键点,坐标缩放到原始图片中
                        float pointX_1 = (point[0] * net_stride[index] + anchorX)  * imgScale;
                        float pointY_1 = (point[1] * net_stride[index] + anchorY)  * imgScale;
                        float pointX_2 = (point[2] * net_stride[index] + anchorX)  * imgScale;
                        float pointY_2 = (point[3] * net_stride[index] + anchorY)  * imgScale;
                        float pointX_3 = (point[4] * net_stride[index] + anchorX)  * imgScale;
                        float pointY_3 = (point[5] * net_stride[index] + anchorY)  * imgScale;
                        float pointX_4 = (point[6] * net_stride[index] + anchorX)  * imgScale;
                        float pointY_4 = (point[7] * net_stride[index] + anchorY)  * imgScale;
                        float pointX_5 = (point[8] * net_stride[index] + anchorX)  * imgScale;
                        float pointY_5 = (point[8] * net_stride[index] + anchorY)  * imgScale;
                        // 保存到tmp中,注意x1y1x2y2不能超出 w*h
                        temp.add(new float[]{
                                score,
                                x1>w?w:x1,
                                y1>h?h:y1,
                                x2>w?w:x2,
                                y2>h?h:y2,
                                pointX_1,pointY_1,
                                pointX_2,pointY_2,
                                pointX_3,pointY_3,
                                pointX_4,pointY_4,
                                pointX_5,pointY_5
                        });
                    }
                }
            }

            // nms
            ArrayList<float[]> datas_after_nms = new ArrayList<>();
            while (!temp.isEmpty()){
                float[] max = temp.get(0);
                datas_after_nms.add(max);
                Iterator<float[]> it = temp.iterator();
                while (it.hasNext()) {
                    // nsm阈值
                    float[] obj = it.next();
                    // x1y1x2y2 是 1234
                    double iou = calculateIoU(max,obj);
                    if (iou > iou_thres) {
                        it.remove();
                    }
                }
            }

            // 标注
            datas_after_nms.stream().forEach(n->{

                // 画边框和关键点需要添加偏移
                int x1 = Float.valueOf(n[1]).intValue() + px;
                int y1 = Float.valueOf(n[2]).intValue() + py;
                int x2 = Float.valueOf(n[3]).intValue() + px;
                int y2 = Float.valueOf(n[4]).intValue() + py;
                Imgproc.rectangle(mat, new Point(x1, y1), new Point(x2, y2), green, tickness);

                float point1_x = Float.valueOf(n[5]).intValue() + px;// 关键点1
                float point1_y = Float.valueOf(n[6]).intValue() + py;//
                float point2_x = Float.valueOf(n[7]).intValue() + px;// 关键点2
                float point2_y = Float.valueOf(n[8]).intValue() + py;//
                float point3_x = Float.valueOf(n[9]).intValue() + px;// 关键点3
                float point3_y = Float.valueOf(n[10]).intValue() + py;//

                Imgproc.circle(mat, new Point(point1_x, point1_y), 1, green, tickness);
                Imgproc.circle(mat, new Point(point2_x, point2_y), 1, green, tickness);
                Imgproc.circle(mat, new Point(point3_x, point3_y), 1, green, tickness);

            });


            tensor.close();
            out.close();
        }
        catch (Exception e){
            e.printStackTrace();
        }
    }


    // 计算两个框的交并比
    private static double calculateIoU(float[] box1, float[] box2) {
        //  getXYXY() 返回 xmin-0 ymin-1 xmax-2 ymax-3
        double x1 = Math.max(box1[1], box2[1]);
        double y1 = Math.max(box1[2], box2[2]);
        double x2 = Math.min(box1[3], box2[3]);
        double y2 = Math.min(box1[4], box2[4]);
        double intersectionArea = Math.max(0, x2 - x1 + 1) * Math.max(0, y2 - y1 + 1);
        double box1Area = (box1[3] - box1[1] + 1) * (box1[4] - box1[2] + 1);
        double box2Area = (box2[3] - box2[1] + 1) * (box2[4] - box2[2] + 1);
        double unionArea = box1Area + box2Area - intersectionArea;
        return intersectionArea / unionArea;
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




