package com.sherpa.dmeo.util;

import android.app.Activity;
import android.content.res.AssetManager;
import android.media.MediaPlayer;
import android.util.Log;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

/**
 * @desc : 一些工具
 * @auth : tyf
 * @date : 2024-10-18 11:35:37
*/
public class Tools {

    private static String TAG = Tools.class.getName();

    // 设置 context
    private static Activity context;
    public static void setContext(Activity context){
        Tools.context = context;
    }

    // 递归复制文件
    public static void copyAsset(String assetPath, String root) {
        AssetManager assetManager = context.getAssets(); // 获取 AssetManager
        String[] files = null;
        try {
            // 获取指定目录下的所有文件和目录
            files = assetManager.list(assetPath);
        } catch (IOException e) {
            e.printStackTrace();
        }
        if (files != null) {
            for (String filename : files) {
                String assetFilePath = assetPath + "/" + filename; // 资源文件的完整路径
                String destFilePath = root + "/" + assetPath + "/" + filename; // 目标文件的完整路径
                try {
                    // 判断是否为目录
                    if (assetManager.list(assetFilePath) != null && assetManager.list(assetFilePath).length > 0) {
                        // 如果是目录，则创建目录并递归复制
                        new File(destFilePath).mkdirs();
                        copyAsset(assetFilePath, root);
                    } else {
                        InputStream in = assetManager.open(assetFilePath);
                        OutputStream out = new FileOutputStream(destFilePath);
                        // 复制文件
                        byte[] buffer = new byte[1024];
                        int read;
                        while ((read = in.read(buffer)) != -1) {
                            out.write(buffer, 0, read);
                        }
                        // 关闭流
                        in.close();
                        out.flush();
                        out.close();
                        Log.d(TAG, "文件复制：" + destFilePath + "，成功：" + new File(destFilePath).exists());
                    }
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    // 播放 wav
    public static void play(String wav) {
        MediaPlayer mediaPlayer = new MediaPlayer();
        try {
            mediaPlayer.setDataSource(wav);
            mediaPlayer.setOnPreparedListener(mp -> {
                mp.start(); // 播放音频
            });
            mediaPlayer.setOnCompletionListener(mp -> {
                mp.release(); // 释放资源
            });
            mediaPlayer.prepare(); // 准备播放
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // 获取 app 存储路径
    public static String path(){
        return context.getExternalFilesDir(null) + "";
    }


}
