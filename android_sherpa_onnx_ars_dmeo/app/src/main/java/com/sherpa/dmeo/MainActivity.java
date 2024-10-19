package com.sherpa.dmeo;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.util.Log;

import com.k2fsa.sherpa.onnx.*;
import com.k2fsa.sherpa.onnx.OnlineZipformer2CtcModelConfig;
import com.k2fsa.sherpa.onnx.WaveReader;
import com.sherpa.dmeo.databinding.ActivityMainBinding;
import com.sherpa.dmeo.util.Tools;

import java.io.File;
import java.util.concurrent.Executors;

/**
 * @desc : TTS/ASR 测试
 * @auth : tyf
 * @date : 2024-10-18 10:33:03
*/
public class MainActivity extends AppCompatActivity {

    private ActivityMainBinding binding;

    private static String TAG = MainActivity.class.getName();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        // 动态库下载，引入：
        // https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.10.28/sherpa-onnx-v1.10.28-android.tar.bz2
        // jniLibs.srcDirs = ['src/main/jniLibs', 'src/main/libs']

        // 查看动态库符号（关键字sherpa ）：
        // nm -D libsherpa-onnx-jni.so | grep "sherpa"

        // 接口下载引入：
        // https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.10.28/sherpa-onnx-v1.10.28-java8.jar
        // implementation fileTree(dir: 'libs', include: ['sherpa-onnx-v1.10.28-java8.jar'])


        // 语音合成测试
        Executors.newSingleThreadExecutor().submit(()->{

            // 递归复制模型文件到 app 存储路径
            Tools.setContext(this);

            String asset = "sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13";
            Tools.copyAsset(asset,Tools.path());

            // 模型
            String model = Tools.path() + "/" + asset+ "/ctc-epoch-20-avg-1-chunk-16-left-128.onnx";
            String tokens = Tools.path() + "/" + asset +"/tokens.txt";

            // wav 测试文件
            String waveFilename = Tools.path() + "/" + asset + "/test_wavs/DEV_T0000000000.wav";

            Log.d(TAG,"实时语音识别");

            WaveReader reader = new WaveReader(waveFilename);
            OnlineZipformer2CtcModelConfig ctc = OnlineZipformer2CtcModelConfig.builder().setModel(model).build();
            OnlineModelConfig modelConfig = OnlineModelConfig.builder()
                    .setZipformer2Ctc(ctc)
                    .setTokens(tokens)
                    .setNumThreads(1)
                    .setDebug(true)
                    .build();
            OnlineRecognizerConfig config = OnlineRecognizerConfig.builder()
                    .setOnlineModelConfig(modelConfig)
                    .setDecodingMethod("greedy_search")
                    .build();
            OnlineRecognizer recognizer = new OnlineRecognizer(config);

            // 流式识别
            OnlineStream stream = recognizer.createStream();
            stream.acceptWaveform(reader.getSamples(), reader.getSampleRate());
            float[] tailPaddings = new float[(int) (0.3 * reader.getSampleRate())];
            stream.acceptWaveform(tailPaddings, reader.getSampleRate());
            while (recognizer.isReady(stream)) {
                recognizer.decode(stream);
            }
            String text = recognizer.getResult(stream).getText();
            stream.release();
            recognizer.release();

            Log.d(TAG,"识别完成："+waveFilename);
            Log.d(TAG,"识别结果："+text);

        });


    }


}