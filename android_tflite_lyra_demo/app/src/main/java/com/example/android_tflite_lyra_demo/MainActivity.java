package com.example.android_tflite_lyra_demo;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import android.annotation.SuppressLint;
import android.media.AudioFormat;
import android.media.AudioManager;
import android.media.AudioRecord;
import android.media.AudioTrack;
import android.media.MediaRecorder;
import android.os.Bundle;
import android.util.Log;
import android.widget.Button;
import android.widget.Toast;

import com.example.android_tflite_lyra_demo.databinding.ActivityMainBinding;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * @desc : 测试 google lyra
 * @auth : tyf
 * @date : 2025-04-27 10:05:27
 */
public class MainActivity extends AppCompatActivity {

    private static final String TAG = MainActivity.class.getName();
    private ActivityMainBinding binding;

    private File pcmFile;
    private AtomicBoolean isRecording = new AtomicBoolean(false);
    private Thread recordThread;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        // 编码器
        Codec.initModelFile(this);
        Codec.initCodec(this);

        // 检测或申请权限
        String perList[] = new String[]{
                android.Manifest.permission.RECORD_AUDIO,
                android.Manifest.permission.WRITE_EXTERNAL_STORAGE,
                android.Manifest.permission.READ_EXTERNAL_STORAGE,
        };
        ActivityCompat.requestPermissions(this, perList, 1);

        pcmFile = new File(getExternalFilesDir(null), "test.pcm");

        Button recordButton = binding.btnRecord;
        Button playButton = binding.btnPlay;

        // 点击录制按钮
        recordButton.setOnClickListener(v -> {
            if (!isRecording.get()) {
                // 开始录制
                startRecording();
                recordButton.setText("停止");
            } else {
                // 停止录制
                stopRecording();
                recordButton.setText("录制");
            }
        });

        // 点击播放按钮
        playButton.setOnClickListener(v -> {
            if (isRecording.get()) {
                Toast.makeText(this, "录制中不能播放", Toast.LENGTH_SHORT).show();
                return;
            }
            if (!pcmFile.exists()) {
                Toast.makeText(this, "录制文件不存在", Toast.LENGTH_SHORT).show();
                return;
            }
            startPlaying();
        });

    }


    // 开始录制
    private void startRecording() {
        isRecording.set(true);
        recordThread = new Thread(() -> {
            try {
                @SuppressLint("MissingPermission")
                AudioRecord recorder = new AudioRecord(
                        MediaRecorder.AudioSource.VOICE_COMMUNICATION,
                        16000,
                        AudioFormat.CHANNEL_IN_MONO,
                        AudioFormat.ENCODING_PCM_16BIT,
                        640
                );

                recorder.startRecording();
                FileOutputStream fos = new FileOutputStream(pcmFile);
                byte[] buffer = new byte[640];

                while (isRecording.get()) {
                    int read = recorder.read(buffer, 0, buffer.length);
                    if (read > 0) {
                        // 音频数据
                        byte[] data = new byte[read];
                        System.arraycopy(buffer, 0, data, 0, read);
                        // 编码
                        byte[] encode = Codec.encode(data);
                        // 解码
                        byte[] decode = Codec.decode(encode);
                        fos.write(decode);
                    }
                }

                recorder.stop();
                recorder.release();
                fos.close();
                Log.d(TAG, "录制结束，保存到：" + pcmFile.getAbsolutePath());
            } catch (Exception e) {
                e.printStackTrace();
            }
        });
        recordThread.start();
    }

    // 停止录制
    private void stopRecording() {
        isRecording.set(false);
        if (recordThread != null && recordThread.isAlive()) {
            try {
                recordThread.join();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
        Toast.makeText(this, "录制完成", Toast.LENGTH_SHORT).show();
    }

    // 开始播放
    private void startPlaying() {
        new Thread(() -> {
            try {
                AudioTrack audioTrack = new AudioTrack(
                        AudioManager.STREAM_MUSIC,
                        16000,
                        AudioFormat.CHANNEL_OUT_MONO,
                        AudioFormat.ENCODING_PCM_16BIT,
                        AudioTrack.getMinBufferSize(16000, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT),
                        AudioTrack.MODE_STREAM
                );

                FileInputStream fis = new FileInputStream(pcmFile);
                byte[] buffer = new byte[640];

                audioTrack.play();

                int read;
                while ((read = fis.read(buffer)) > 0) {
                    byte[] data = new byte[read];
                    System.arraycopy(buffer, 0, data, 0, read);
                    audioTrack.write(data, 0, data.length);
                }
                audioTrack.stop();
                audioTrack.release();
                fis.close();
                Log.d(TAG, "播放完成");
            } catch (Exception e) {
                e.printStackTrace();
            }
        }).start();
    }
}
