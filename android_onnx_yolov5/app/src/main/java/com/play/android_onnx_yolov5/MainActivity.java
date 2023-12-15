package com.play.android_onnx_yolov5;

import ai.onnxruntime.*;
import android.content.ContentResolver;
import android.content.Intent;
import android.content.res.AssetManager;
import android.graphics.*;
import android.net.Uri;
import android.provider.MediaStore;
import android.view.Display;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.appcompat.app.AppCompatActivity;
import android.os.Bundle;
import com.alibaba.fastjson.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.*;

public class MainActivity extends AppCompatActivity {


    // onnxruntime 环境
    public static OrtEnvironment env;
    public static OrtSession session;

    // 显示推理耗时
    private TextView timeuse;

    // 图片识别结果显示
    private ImageView imageView;

    // 按钮用于选择本地图片
    private Button button;

    // 标注边框和文本的画笔
    Paint paint1 = new Paint();
    Paint paint2 = new Paint();

    // 类别信息
    static List<String> clazzStr;

    @Override
    protected void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // 显示推理耗时
        timeuse = findViewById(R.id.timeuse);

        // 加载onnx模型
        this.loadOnnx();

        // 按钮用于选择本地图片
        button = findViewById(R.id.openpic);
        button.setOnClickListener(v -> openGallery());

        // 图片区域显示本地图片识别结果,默认黑色背景
        imageView = findViewById(R.id.image);
        Display display = this.getWindowManager().getDefaultDisplay();
        ViewGroup.LayoutParams layoutParams = imageView.getLayoutParams();
        layoutParams.width = display.getWidth();
        layoutParams.height = Double.valueOf(Double.valueOf(display.getWidth())*1.3).intValue();
        imageView.setLayoutParams(layoutParams);

        // 标注边框的画笔
        paint1.setStyle(Paint.Style.STROKE);
        paint1.setColor(Color.RED);
        paint1.setStrokeWidth(2.5f);
        // 标注文本的画笔,设置本地字体
        paint2.setColor(Color.GREEN);
        paint2.setTextSize(13);
        paint2.setAntiAlias(true);
        paint2.setFakeBoldText(true);
        paint2.setStyle(Paint.Style.FILL);
        paint2.setTextAlign(Paint.Align.LEFT);

    }


    // 加载onnx模型
    public void loadOnnx() {

        try {
            // 获取 AssetManager 对象来访问 src/main/assets 目录,读取模型的字节数组
            AssetManager assetManager = getAssets();
            InputStream inputStream = assetManager.open("yolov5s.onnx");
            ByteArrayOutputStream buffer = new ByteArrayOutputStream();
            int nRead;
            byte[] data = new byte[1024];
            while ((nRead = inputStream.read(data, 0, data.length)) != -1) {
                buffer.write(data, 0, nRead);
            }
            buffer.flush();
            byte[] module = buffer.toByteArray();
            // images -> [1, 3, 640, 640] -> FLOAT
            // output0 -> [1, 25200, 85] -> FLOAT
            System.out.println("开始加载模型");
            env = OrtEnvironment.getEnvironment();
            session = env.createSession(module, new OrtSession.SessionOptions());
            // 打印模型信息,获取输入输出的shape以及类型：
            System.out.println("模型输入:");
            session.getInputInfo().entrySet().stream().forEach(n -> {
                String inputName = n.getKey();
                NodeInfo inputInfo = n.getValue();
                long[] shape = ((TensorInfo) inputInfo.getInfo()).getShape();
                String javaType = ((TensorInfo) inputInfo.getInfo()).type.toString();
                System.out.println(inputName + " -> " + Arrays.toString(shape) + " -> " + javaType);
            });
            System.out.println("模型输出:");
            session.getOutputInfo().entrySet().stream().forEach(n -> {
                String outputName = n.getKey();
                NodeInfo outputInfo = n.getValue();
                long[] shape = ((TensorInfo) outputInfo.getInfo()).getShape();
                String javaType = ((TensorInfo) outputInfo.getInfo()).type.toString();
                System.out.println(outputName + " -> " + Arrays.toString(shape) + " -> " + javaType);
            });
            // 类别信息
            JSONObject clazz = JSONObject.parseObject(session.getMetadata().getCustomMetadata().get("names").replace("\"","\"\""));
            clazzStr = new ArrayList<>();
            clazz.entrySet().forEach(n->{
                clazzStr.add(String.valueOf(n.getValue()));
            });
            System.out.println("类别信息:"+clazzStr);
        } catch (Exception e) {
            e.printStackTrace();
        }

    }


    // 选择相册图片
    public void openGallery(){
        Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        startActivityForResult(intent, 1);
    }


    // 返回选择的图片
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == 1 && resultCode == RESULT_OK && data != null) {
            Uri imageUri = data.getData();
            // 处理选择的图片
            task(imageUri);
        }
    }



    // 对选择的图片进行处理
    private Bitmap task(Uri imageUri){

        // 处理结果
        Bitmap copy = null;

        try {

            // 获取图片输入流,转为bitmap
            ContentResolver contentResolver = this.getContentResolver();
            InputStream inputStream = contentResolver.openInputStream(imageUri);
            Bitmap bitmap = BitmapFactory.decodeStream(inputStream);

            // 上面bitmap是本地文件不可变,这里创建一个副本来处理
            copy = bitmap.copy(Bitmap.Config.ARGB_8888, true);

            // 推理
            long t1 = System.currentTimeMillis();
            copy = inferencr(copy);
            long t2 = System.currentTimeMillis();

            // 显示标注后的图片
            imageView.setImageBitmap(copy);

            // 显示耗时
            timeuse.setText("推理耗时："+(t2-t1)+"ms");

        }
        catch (Exception e){
            e.printStackTrace();
        }

        // 返回可变图片对象
        return copy;
    }



    // 图片预处理推理和后处理
    private Bitmap inferencr(Bitmap copy){

        int w = 640;
        int h = 640;
        int c = 3;

        // 修改图像尺寸
        Bitmap bitmap = Bitmap.createScaledBitmap(copy,w,h,false);

        // 提取rgb(chw存储)并做归一化
        float[] rgb = new float[c*h*w];

        // 从bitmap中提取像素数据,每个位置可以分别计算rgb三个分量
        int[] pixels = new int[w*h];
        bitmap.getPixels(pixels, 0, w, 0, 0, w, h);

        // chw的排放在一维数组中是这样的,以5个像素点为例
        // rrrrr ggggg bbbbb

        // 遍历每个像素点 w*h个
        for (int i = 0; i < pixels.length; i++) {
            int pixel = pixels[i];
            // rgb分量归一化处理
            float r = ((pixel >> 16) & 0xFF) / 255f;
            float g = ((pixel >> 8) & 0xFF) / 255f;
            float b = (pixel & 0xFF) / 255f;
            // 存储到一维数组中
            rgb[i] = r;
            rgb[i+(w*h)] = g;
            rgb[i+(w*h)+(w*h)] = b;
        }

        // 创建张量并进行推理
        try {
            OnnxTensor tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(rgb), new long[]{1,c,h,w});
            System.out.println("张量创建成功");
            OrtSession.Result output = session.run(Collections.singletonMap("images", tensor));
            System.out.println("推理成功");

            OnnxTensor tensor_out = (OnnxTensor)output.get(0);
            float[][] data = ((float[][][])tensor_out.getValue())[0];

            List<Detection> detections = new ArrayList<>();

            // 25200
            for (float[] box : data){
                // 置信度
                if(box[4]>0.5f){
                    detections.add(new Detection(box));
                }
            }

            // nms
            List<Detection> detections_after_nms = new ArrayList<>();
            detections.sort((o1, o2) -> Float.compare(o2.type_max_value,o1.type_max_value));
            while (!detections.isEmpty()){
                Detection maxObj = detections.get(0);
                detections_after_nms.add(maxObj);
                Iterator<Detection> it = detections.iterator();
                while (it.hasNext()) {
                    Detection obj = it.next();
                    // 计算交并比
                    if(Detection.calculateIoU(maxObj,obj)>0.45f){
                        it.remove();
                    }
                }
            }

            // 标注边框和类别
            Canvas canvas = new Canvas(bitmap);
            detections_after_nms.stream().forEach(n->{
                canvas.drawLine(n.x1,n.y1,n.x2,n.y1, paint1);
                canvas.drawLine(n.x1,n.y2,n.x2,n.y2, paint1);
                canvas.drawLine(n.x1,n.y1,n.x1,n.y2, paint1);
                canvas.drawLine(n.x2,n.y1,n.x2,n.y2, paint1);
                canvas.drawText(n.type_max_name, n.x1, n.y1, paint2);
            });

        }
        catch (Exception e){
            e.printStackTrace();
        }

        return bitmap;
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

}