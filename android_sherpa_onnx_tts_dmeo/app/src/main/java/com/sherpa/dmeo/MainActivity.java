package com.sherpa.dmeo;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.util.Log;

import com.k2fsa.sherpa.onnx.GeneratedAudio;
import com.k2fsa.sherpa.onnx.OfflineTts;
import com.k2fsa.sherpa.onnx.OfflineTtsConfig;
import com.k2fsa.sherpa.onnx.OfflineTtsModelConfig;
import com.k2fsa.sherpa.onnx.OfflineTtsVitsModelConfig;
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
            Tools.copyAsset("vits-melo-tts-zh_en",Tools.path());

            String model = Tools.path() + "/vits-melo-tts-zh_en/model.onnx";
            String tokens = Tools.path() + "/vits-melo-tts-zh_en/tokens.txt";
            String lexicon = Tools.path() + "/vits-melo-tts-zh_en/lexicon.txt";
            String dictDir = Tools.path() + "/vits-melo-tts-zh_en/dict";
            String ruleFsts = Tools.path() + "/vits-melo-tts-zh_en/phone.fst," +
                    Tools.path() + "/vits-melo-tts-zh_en/date.fst," +
                    Tools.path() +"/vits-melo-tts-zh_en/number.fst," +
                    Tools.path() +"/vits-melo-tts-zh_en/new_heteronym.fst";

            // 待生成文本
            String text = "在晨光初照的时分，\n" +
                    "微风轻拂，花瓣轻舞，\n" +
                    "小溪潺潺，诉说心事，\n" +
                    "阳光透过树梢，洒下温暖。\n" +
                    "\n" +
                    "远山如黛，静默守望，\n" +
                    "白云悠悠，似梦似幻，\n" +
                    "时光流转，岁月如歌，\n" +
                    "愿心中永存这份宁静。\n" +
                    "\n" +
                    "无论何时，心怀希望，\n" +
                    "在每一个晨曦中起舞，\n" +
                    "追逐梦想，勇往直前，\n" +
                    "让生命绽放出灿烂的光彩。";

            // 输出wav文件
            String waveFilename = Tools.path() + "/tts-vits-zh.wav";

            Log.d(TAG,"开始语音合成!");

            OfflineTtsVitsModelConfig vitsModelConfig = OfflineTtsVitsModelConfig.builder()
                    .setModel(model)
                    .setTokens(tokens)
                    .setLexicon(lexicon)
                    .setDictDir(dictDir)
                    .build();

            OfflineTtsModelConfig modelConfig = OfflineTtsModelConfig.builder()
                    .setVits(vitsModelConfig)
                    .setNumThreads(1)
                    .setDebug(true)
                    .build();

            OfflineTtsConfig config = OfflineTtsConfig.builder().setModel(modelConfig).setRuleFsts(ruleFsts).build();
            OfflineTts tts = new OfflineTts(config);

            // 语速和说话人
            int sid = 100;
            float speed = 1.0f;
            long start = System.currentTimeMillis();
            GeneratedAudio audio = tts.generate(text, sid, speed);
            long stop = System.currentTimeMillis();
            float timeElapsedSeconds = (stop - start) / 1000.0f;
            float audioDuration = audio.getSamples().length / (float) audio.getSampleRate();
            float real_time_factor = timeElapsedSeconds / audioDuration;

            audio.save(waveFilename);
            Log.d(TAG, String.format("-- elapsed : %.3f seconds", timeElapsedSeconds));
            Log.d(TAG, String.format("-- audio duration: %.3f seconds", timeElapsedSeconds));
            Log.d(TAG, String.format("-- real-time factor (RTF): %.3f", real_time_factor));
            Log.d(TAG, String.format("-- text: %s", text));
            Log.d(TAG, String.format("-- Saved to %s", waveFilename));

            Log.d(TAG,"音频合成："+waveFilename+"，是否成功："+new File(waveFilename).exists());
            tts.release();

            // 播放 wav
            Tools.play(waveFilename);

        });


    }


}