package demo.android_onnx_maskrcnn;

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

    Paint paint = new Paint();

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
        paint.setStyle(Paint.Style.STROKE);
        paint.setColor(Color.RED);
        paint.setStrokeWidth(2.5f);

    }


    // 加载onnx模型
    public void loadOnnx() {

        try {
            // 获取 AssetManager 对象来访问 src/main/assets 目录,读取模型的字节数组

            // 因为模型比较大需要加入下面配置
            // android:hardwareAccelerated = "false"
            // android:largeHeap = "true"

            AssetManager assetManager = getAssets();
            InputStream inputStream = assetManager.open("maskrcnn_resnet50_fpn.onnx");
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
            Bitmap output = inferencr(copy);
            long t2 = System.currentTimeMillis();
            // 显示标注后的图片
            imageView.setImageBitmap(output);
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

        // 模型输入宽高,onnx网站可以看到
        int inputCount = 1;
        int inputChannel = 3;
        int inputWidth = 1024;
        int inputHeight = 1024;

        Bitmap bitmap = Bitmap.createScaledBitmap(copy,inputWidth,inputHeight,false);

        // 提取rgb(chw存储)并做归一化
        float[] rgb = new float[ inputChannel * inputWidth * inputHeight ];

        // 从bitmap中提取像素数据,每个位置可以分别计算rgb三个分量
        int[] pixels = new int[ inputWidth * inputHeight ];
        bitmap.getPixels(pixels, 0, inputWidth, 0, 0, inputWidth, inputHeight);

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
            rgb[i+(inputWidth*inputHeight)] = g;
            rgb[i+(inputWidth*inputHeight)+(inputWidth*inputHeight)] = b;
        }

        try {

            OnnxTensor tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(rgb), new long[]{inputCount,inputChannel,inputHeight,inputWidth});
            System.out.println("张量创建成功");
            OrtSession.Result output = session.run(Collections.singletonMap("images", tensor));

            // 获取四个输出
            OnnxTensor boxes = (OnnxTensor)output.get("boxes").get();
            OnnxTensor labels = (OnnxTensor)output.get("labels").get();
            OnnxTensor scores = (OnnxTensor)output.get("scores").get();
            OnnxTensor masks = (OnnxTensor)output.get("masks").get();

            // 获取输出的张量
            float[][] boxesArray = (float[][])boxes.getValue();
            long[] labelsArray = (long[])labels.getValue();
            float[] scoresArray = (float[])scores.getValue();
            float[][][][] masksArray = (float[][][][])masks.getValue();

            // 目标个数
            int size = labelsArray.length;
            // 遍历每个目标,生成目标对象
            List<Obj> objs = new ArrayList<>();
            for (int i=0;i<size;i++){
                float[] xyxy = boxesArray[i];// 边框信息
                long label = labelsArray[i];// 类别信息
                float score = scoresArray[i];// 得分
                float[][][] mask = masksArray[i];// 掩膜信息
                objs.add(new Obj(xyxy,label,score,mask));
            }

            // 删除得分低的
            objs.removeIf(obj -> obj.score<0.3f);

            // 先按得分降序
            objs.sort((o1, o2) -> Float.compare(o2.score,o1.score));

            // nms过滤之后的
            List<Obj> nms = new ArrayList<>();
            while (!objs.isEmpty()){
                Obj max = objs.get(0);
                nms.add(max);
                Iterator<Obj> it = objs.iterator();
                while (it.hasNext()){
                    Obj bi = it.next();
                    // 计算交并比
                    if(calculateIoU(max.xyxy,bi.xyxy)>=0.45f){
                        it.remove();
                    }
                }
            }

            // 标注边框和前景
            Canvas canvas = new Canvas(bitmap);
            nms.stream().forEach(n->{
                // 标注边框
                float[] box = n.xyxy;
                int xmin = Float.valueOf(box[0]).intValue();
                int ymin = Float.valueOf(box[1]).intValue();
                int xmax = Float.valueOf(box[2]).intValue();
                int ymax = Float.valueOf(box[3]).intValue();
                canvas.drawLine(xmin,ymin,xmax,ymin,paint);
                canvas.drawLine(xmin,ymax,xmax,ymax,paint);
                canvas.drawLine(xmin,ymin,xmin,ymax,paint);
                canvas.drawLine(xmax,ymin,xmax,ymax,paint);
                // 标注mask前景 mask掩膜 1 * netWidth * netHeight 这里直接取第一个
                float[][] mask = n.mask[0];
                for (int y = ymin; y <= ymax; y++) {
                    for (int x = xmin; x <= xmax; x++) {
                        float pixelValue = mask[y][x];
                        // 大于阈值,说明这个点可能是目标 x,y
                        if(pixelValue > 0.5f){
                            // 修改 bitmap 这个店的颜色
                            int color = bitmap.getPixel(x,y);
                            int r = Color.red(color);
                            int g = Color.green(color);
                            int b = Color.blue(color);
                            int a = Color.alpha(color);
                            // 修改透明度和颜色
                            b = 255;
                            bitmap.setPixel(x,y,Color.argb(a,r,g,b));
                        }
                    }
                }
            });
        }
        catch (Exception e){
            e.printStackTrace();
        }
        System.out.println("推理完成!");
        return bitmap;
    }

    // 目标对象
    public static class Obj{
        float[] xyxy;// 边框信息
        long label;// 类别信息
        float score;// 得分
        float[][][] mask;// 掩膜信息
        List<int[]> point;// mask 按照阈值过滤后保留的点
        public Obj(float[] xyxy, long label, float score, float[][][] mask) {
            this.xyxy = xyxy;
            this.label = label;
            this.score = score;
            this.mask = mask;
        }

        public void setPoint(List<int[]> point) {
            this.point = point;
        }
    }

    // 交并比
    private static double calculateIoU(float[] box1, float[] box2) {
        double x1 = Math.max(box1[0], box2[0]);
        double y1 = Math.max(box1[1], box2[1]);
        double x2 = Math.min(box1[2], box2[2]);
        double y2 = Math.min(box1[3], box2[3]);
        double intersectionArea = Math.max(0, x2 - x1 + 1) * Math.max(0, y2 - y1 + 1);
        double box1Area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1);
        double box2Area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1);
        double unionArea = box1Area + box2Area - intersectionArea;
        return intersectionArea / unionArea;
    }

}