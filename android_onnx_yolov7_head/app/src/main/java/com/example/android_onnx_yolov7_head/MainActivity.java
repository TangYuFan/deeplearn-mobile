package com.example.android_onnx_yolov7_head;

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

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.Collections;

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

    @Override
    protected void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // 加载onnx模型
        this.loadOnnx();

        // 显示推理耗时
        timeuse = findViewById(R.id.timeuse);

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
        paint1.setStrokeWidth(3);
        // 标注文本的画笔,设置本地字体
        paint2.setColor(Color.RED);
        paint2.setTextSize(15);
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
            InputStream inputStream = assetManager.open("yolov7_tiny_head_0.768_post_480x640.onnx");
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
        int h = 480;
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
            OrtSession.Result output = session.run(Collections.singletonMap("input", tensor));
            System.out.println("推理成功");

            // 解析输出,处理目标框(模型做了合并nms处理)
            long[][] batchno_classid_y1x1y2x2 = ((long[][])(output.get(1)).getValue());
            int num = batchno_classid_y1x1y2x2.length;// 检测到的目标个数
            System.out.println("检测到目标个数:"+num);
            for(int i=0;i<num;i++){

                long[] data = batchno_classid_y1x1y2x2[i];
                float y1 = data[2];
                float x1 = data[3];
                float y2 = data[4];
                float x2 = data[5];


                // 标注边框
                Canvas canvas = new Canvas(bitmap);
                canvas.drawLine(x1,y1,x2,y1, paint1);
                canvas.drawLine(x1,y2,x2,y2, paint1);
                canvas.drawLine(x1,y1,x1,y2, paint1);
                canvas.drawLine(x2,y1,x2,y2, paint1);
                // 标注类别
                canvas.drawText("人头", x1, y1, paint2);

            }

        }
        catch (Exception e){
            e.printStackTrace();
        }

        return bitmap;
    }

}