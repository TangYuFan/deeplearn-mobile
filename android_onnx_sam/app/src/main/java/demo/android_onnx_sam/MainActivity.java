package demo.android_onnx_sam;

import ai.onnxruntime.*;
import android.content.ContentResolver;
import android.content.Intent;
import android.content.res.AssetManager;
import android.graphics.*;
import android.net.Uri;
import android.provider.MediaStore;
import android.view.MotionEvent;
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
import java.util.concurrent.atomic.AtomicBoolean;

public class MainActivity extends AppCompatActivity {


    // onnxruntime 环境
    public static OrtEnvironment env1;
    public static OrtSession session1;
    public static OrtEnvironment env2;
    public static OrtSession session2;

    // 显示推理耗时
    private TextView timeuse;

    // 图片识别结果显示 1024,1024区域
    private ImageView imageView;
    // 抠图点,1024,1024区域内
    private List<float[]> points;


    // 当前显示的图片的备份
    private Bitmap copy;

    // 按钮用于选择本地图片
    private Button openpic;
    private Button koupic;


    @Override
    protected void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // 显示推理耗时
        timeuse = findViewById(R.id.timeuse);

        // 加载onnx模型
        this.loadOnnx1("encoder-vit_b.quant.onnx");
        this.loadOnnx2("decoder-vit_b.quant.onnx");

        // 按钮用于选择本地图片
        openpic = findViewById(R.id.openpic);
        openpic.setOnClickListener(v -> openGallery());

        // 按钮用于开始抠图
        koupic = findViewById(R.id.koupic);
        koupic.setOnClickListener(v-> task());

        // 图片区域显示本地图片识别结果,默认黑色背景
        imageView = findViewById(R.id.image);
        ViewGroup.LayoutParams layoutParams = imageView.getLayoutParams();
        layoutParams.width = 1024;
        layoutParams.height = 1024;
        imageView.setLayoutParams(layoutParams);

        // 图片区域监听点击事件,收集抠图点
        points = new ArrayList<>();
        imageView.setOnTouchListener((v, event) -> {
            if(copy==null){
                return false;
            }
            // 如果是按下
            if(MotionEvent.ACTION_DOWN==event.getAction()){
                float x = event.getX();
                float y = event.getY();
                System.out.println("坐标点搜集:"+x+","+y+",默认背景");
                points.add(new float[]{x,y,1});
                // 将这个点在bitmap中画出来
                Bitmap show = Bitmap.createScaledBitmap(copy,1024,1024,false);
                Canvas canvas = new Canvas(show);
                Paint pai = new Paint();
                pai.setStyle(Paint.Style.STROKE);
                pai.setColor(Color.RED);
                pai.setStrokeWidth(20f);
                points.stream().forEach(n->{
                    canvas.drawPoint(n[0],n[1],pai);
                });
                imageView.setImageBitmap(show);
            }
            return false;
        });

    }


    // 加载onnx模型
    public void loadOnnx1(String modulePath) {

        try {
            // 获取 AssetManager 对象来访问 src/main/assets 目录,读取模型的字节数组
            AssetManager assetManager = getAssets();
            InputStream inputStream = assetManager.open(modulePath);
            ByteArrayOutputStream buffer = new ByteArrayOutputStream();
            int nRead;
            byte[] data = new byte[1024];
            while ((nRead = inputStream.read(data, 0, data.length)) != -1) {
                buffer.write(data, 0, nRead);
            }
            buffer.flush();
            byte[] module = buffer.toByteArray();
            System.out.println("开始加载模型");
            env1 = OrtEnvironment.getEnvironment();
            session1 = env1.createSession(module, new OrtSession.SessionOptions());
            // 打印模型信息,获取输入输出的shape以及类型：
            session1.getInputInfo().entrySet().stream().forEach(n -> {
                String inputName = n.getKey();
                NodeInfo inputInfo = n.getValue();
                long[] shape = ((TensorInfo) inputInfo.getInfo()).getShape();
                String javaType = ((TensorInfo) inputInfo.getInfo()).type.toString();
                System.out.println("模型输入:"+inputName + " -> " + Arrays.toString(shape) + " -> " + javaType);
            });
            session1.getOutputInfo().entrySet().stream().forEach(n -> {
                String outputName = n.getKey();
                NodeInfo outputInfo = n.getValue();
                long[] shape = ((TensorInfo) outputInfo.getInfo()).getShape();
                String javaType = ((TensorInfo) outputInfo.getInfo()).type.toString();
                System.out.println("模型输出:"+outputName + " -> " + Arrays.toString(shape) + " -> " + javaType);
            });
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    public void loadOnnx2(String modulePath) {

        try {
            // 获取 AssetManager 对象来访问 src/main/assets 目录,读取模型的字节数组
            AssetManager assetManager = getAssets();
            InputStream inputStream = assetManager.open(modulePath);
            ByteArrayOutputStream buffer = new ByteArrayOutputStream();
            int nRead;
            byte[] data = new byte[1024];
            while ((nRead = inputStream.read(data, 0, data.length)) != -1) {
                buffer.write(data, 0, nRead);
            }
            buffer.flush();
            byte[] module = buffer.toByteArray();
            System.out.println("开始加载模型");
            env2 = OrtEnvironment.getEnvironment();
            session2 = env2.createSession(module, new OrtSession.SessionOptions());
            // 打印模型信息,获取输入输出的shape以及类型：
            session2.getInputInfo().entrySet().stream().forEach(n -> {
                String inputName = n.getKey();
                NodeInfo inputInfo = n.getValue();
                long[] shape = ((TensorInfo) inputInfo.getInfo()).getShape();
                String javaType = ((TensorInfo) inputInfo.getInfo()).type.toString();
                System.out.println("模型输入:"+inputName + " -> " + Arrays.toString(shape) + " -> " + javaType);
            });
            session2.getOutputInfo().entrySet().stream().forEach(n -> {
                String outputName = n.getKey();
                NodeInfo outputInfo = n.getValue();
                long[] shape = ((TensorInfo) outputInfo.getInfo()).getShape();
                String javaType = ((TensorInfo) outputInfo.getInfo()).type.toString();
                System.out.println("模型输出:"+outputName + " -> " + Arrays.toString(shape) + " -> " + javaType);
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
            try {
                // 清空上次选择的图片和抠图点
                copy = null;
                points.clear();
                // 显示选择的图片
                // 获取图片输入流,转为bitmap
                ContentResolver contentResolver = this.getContentResolver();
                InputStream inputStream = contentResolver.openInputStream(imageUri);
                Bitmap bitmap = BitmapFactory.decodeStream(inputStream);
                // 上面bitmap是本地文件不可变,这里创建一个副本来处理
                Bitmap show = Bitmap.createScaledBitmap(bitmap,1024,1024,false);
                imageView.setImageBitmap(show);
                // 保存备份
                copy = show.copy(Bitmap.Config.ARGB_8888,true);
            }
            catch (Exception e){
                e.printStackTrace();
            }
        }
    }

    // 对选择的图片进行处理
    private static AtomicBoolean doing = new AtomicBoolean(false);
    private void task(){
        if(copy==null||points.size()==0){
            System.out.println("未选择图片或者未选择抠图点");
            return;
        }
        // 正在执行
        if(doing.get()){
            System.out.println("正在抠图中");
            return;
        }
        else{
            doing.set(true);
            System.out.println("开始抠图");
            try {
                // 推理
                long t1 = System.currentTimeMillis();
                Bitmap out = inferencr(copy,points);
                long t2 = System.currentTimeMillis();
                // 显示标注后的图片
                imageView.setImageBitmap(out);
                // 显示耗时
                timeuse.setText("推理耗时："+(t2-t1)+"ms");
            }
            catch (Exception e){
                e.printStackTrace();
            }
            doing.set(false);
            System.out.println("抠图完成");
        }
    }



    // 图片预处理推理和后处理
    private Bitmap inferencr(Bitmap copy,List<float[]> points){
        // 模型1参数
        int module_1_c = 3;
        int module_1_h = 1024;
        int module_1_w = 1024;
        // 图片备份
        Bitmap bitmap = Bitmap.createScaledBitmap(copy,1024,1024,false);
        // chw 排放
        float[] rgb = new float[ module_1_c * module_1_h * module_1_w ];
        int[] pixels = new int[ module_1_w * module_1_h ];
        bitmap.getPixels(pixels, 0, module_1_w, 0, 0, module_1_w, module_1_h);
        for (int i = 0; i < pixels.length; i++) {
            int pixel = pixels[i];
            // rgb分量归一化处理
            float r = ((pixel >> 16) & 0xFF) / 255f;
            float g = ((pixel >> 8) & 0xFF) / 255f;
            float b = (pixel & 0xFF) / 255f;
            // 存储到一维数组中
            rgb[i] = r;
            rgb[i+(module_1_w*module_1_h)] = g;
            rgb[i+(module_1_w*module_1_h)+(module_1_w*module_1_h)] = b;
        }
        // 计算均值和标准差
        float mean = 0.0f;
        float std = 0.0f;
        for (int i = 0; i < rgb.length; i++) {
            mean += rgb[i];
        }
        mean /= rgb.length;
        // 每个元素再进行归一化
        for (int i = 0; i < rgb.length; i++) {
            std += Math.pow(rgb[i] - mean, 2);
        }
        std = (float) Math.sqrt(std / rgb.length);

        // 再将所有元素减去均值并除标准差
        for (int i = 0; i < rgb.length; i++) {
            rgb[i] = (rgb[i] - mean) / std;
        }

        try {
            // 推理1
            OnnxTensor tensor = OnnxTensor.createTensor(env1, FloatBuffer.wrap(rgb), new long[]{1,module_1_c,module_1_h,module_1_w});
            OrtSession.Result res = session1.run(Collections.singletonMap("x", tensor));
            // 输出 256 * 64 * 64
            float[][][] image_embeddings  = ((float[][][][])(res.get(0)).getValue())[0];
            // 推理2
            // 3维转1维 [1, 256, 64, 64]
            float[] chw = this.chw2chw(image_embeddings,256,64,64);
            OnnxTensor _image_embeddings = OnnxTensor.createTensor(env2, FloatBuffer.wrap(chw), new long[]{1,256, 64, 64});
            // point_coords 抠图点 [1, -1, 2]
            float[] pc = new float[points.size()*2];
            // point_labels 类别 [1, -1] 每个抠图点需要设置属于前景还是背景
            float[] pc_label = new float[points.size()];
            for(int i=0;i<points.size();i++){
                // 需要从原始坐标转换到 1024*1024 坐标系中
                float[] xyl = points.get(i);
                pc[i*2] = xyl[0];
                pc[i*2+1] = xyl[1];
                pc_label[i] = xyl[2];
            }
            // 提示点
            OnnxTensor _point_coords = OnnxTensor.createTensor(env2, FloatBuffer.wrap(pc), new long[]{1,points.size(),2});
            OnnxTensor _point_labels = OnnxTensor.createTensor(env2, FloatBuffer.wrap(pc_label), new long[]{1,points.size()});
            //  orig_im_size 原始图像尺寸 [2]
            OnnxTensor _orig_im_size = OnnxTensor.createTensor(env2, FloatBuffer.wrap(new float[]{1024,1024}), new long[]{2});
            // has_mask_input 输入是否包含mask [1]
            OnnxTensor _has_mask_input = OnnxTensor.createTensor(env2, FloatBuffer.wrap(new float[]{0}), new long[]{1});
            // mask_input 提示mask [1, 1, 256, 256] 因为设置了不包含mask这里生成一个固定长度无用的数组即可
            // 这里模型的输出可以作为下一次预测的输入,以提高提示准确性
            float[] ar_256_156 = new float[256*256];
            for(int i=0;i<256*156;i++){
                ar_256_156[i] = 0;
            }
            OnnxTensor _mask_input = OnnxTensor.createTensor(env2, FloatBuffer.wrap(ar_256_156), new long[]{1,1,256,256});
            // 封装参数
            Map<String,OnnxTensor> in = new HashMap<>();
            in.put("image_embeddings",_image_embeddings);
            in.put("point_coords", _point_coords);
            in.put("point_labels",_point_labels);
            in.put("has_mask_input",_has_mask_input);
            in.put("orig_im_size",_orig_im_size);
            in.put("mask_input",_mask_input);
            // 推理
            OrtSession.Result output = session2.run(in);
            // ---------模型2输出-----------
            // masks -> [-1, -1, -1, -1] -> FLOAT
            // iou_predictions -> [-1, 1] -> FLOAT
            // low_res_masks -> [-1, 1, -1, -1] -> FLOAT
            float[][][] masks  = ((float[][][][])(output.get(0)).getValue())[0];
            // 遍历每个 mask 分数最高的排最前面,那么只获取第零个即可
            int count = masks.length;
            float[][] info = null;
            for(int i=0;i < count;i++){
                // 这里输出的每个mask都是模型输入时 orig_im_size 指定的宽高 1024*1024
                // 分数最高的排最前面,那么只获取第零个即可
                info = masks[i];
                break;
            }

            // 将抠出的图片进行渲染 image_3_1024_1024 上进行mask标注
            if(info!=null){
                int sub_w = info.length;
                int sub_h = info[0].length;
                for(int j=0;j<sub_w;j++){
                    for(int k=0;k<sub_h;k++){
                        float da = info[j][k] + 1;
                        if(da>0.5){
                            // 修改颜色为绿色
                            int color = bitmap.getPixel(k,j);
                            int r = Color.red(color);
                            int g = Color.green(color);
                            int b = Color.blue(color);
                            int a = Color.alpha(color);
                            // 修改透明度和颜色
                            b = 255;
                            bitmap.setPixel(k,j,Color.argb(a,r,g,b));
                        }
                    }
                }
            }

            // 显示抠图点
            Canvas canvas = new Canvas(bitmap);
            Paint pai = new Paint();
            pai.setStyle(Paint.Style.STROKE);
            pai.setColor(Color.RED);
            pai.setStrokeWidth(20f);
            points.stream().forEach(n->{
                canvas.drawPoint(n[0],n[1],pai);
            });
            // 推理成功
            System.out.println("推理成功!");
        }
        catch (Exception e){
            e.printStackTrace();
        }

        return bitmap;
    }


    // 3维转1维
    public float[] chw2chw(float[][][] chw,int c,int h,int w){
        float[] res = new float[ c * h * w ];

        int index = 0;
        for(int i=0;i<c;i++){
            for(int j=0;j<h;j++){
                for(int k=0;k<w;k++){
                    float d = chw[i][j][k];
                    res[index] = d;
                    index++;
                }
            }
        }
        return res;
    }


}