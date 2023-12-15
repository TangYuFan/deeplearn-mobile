// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

package camera.demo;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.graphics.PixelFormat;
import android.os.Bundle;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.WindowManager;
import android.widget.Button;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import ncnn.v5lite.demo.R;

import java.util.concurrent.atomic.AtomicBoolean;


/**
*   @desc : opencv + ncnn + ndk 实时预览处理
*   @auth : tyf
*   @date : 2023-07-20  10:18:53
*/
public class MainActivity extends Activity{

    // 预览画面显示
    private SurfaceView surfaceView;
    private AtomicBoolean cameraId = new AtomicBoolean(true);

    // JNI
    private Woker woker = new Woker();

    @Override
    public void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // 屏幕常亮
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        // 隐藏上方状态栏
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN, WindowManager.LayoutParams.FLAG_FULLSCREEN);


        // 按钮,切换摄像头
        Button button =  findViewById(R.id.button);
        button.setOnClickListener(view -> {
            // 线关闭摄像头
            woker.closeCamera();
            // 0前置,1后置
            cameraId.set(!cameraId.get());
            // 再打开摄像头
            woker.openCamera(cameraId.get()?1:0);
        });

        // 加载模型,ndk从assets下加载模型,支持的尺寸如下
        // v5lite-e(bin/param)  320 416
        // v5lite-i8e(bin/param)    320 416
        // v5lite-s(bin/param)  416
        // v5lite-s-int8(bin/param) 418
        woker.loadModule(getAssets());

        // surfaceView + camera2 使用ndk打开相机
        if (ContextCompat.checkSelfPermission(getApplicationContext(), Manifest.permission.CAMERA) == PackageManager.PERMISSION_DENIED) {
            ActivityCompat.requestPermissions(this, new String[] {Manifest.permission.CAMERA}, 100);
        }
        surfaceView = findViewById(R.id.surfaceView);
        surfaceView.getHolder().setFormat(PixelFormat.RGBA_8888);
        surfaceView.getHolder().addCallback(new SurfaceHolder.Callback() {
            @Override
            public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
                // 向ndk传入 surface 使用 ndk camera2渲染帧
                woker.setOutputWindow(holder.getSurface());

                // 这一部分由通讯 ncnn 封装的 ndkcamera 实现调用顺序如下:
                // ndkcamera.cpp nv21_croprotated() =>
                // woker.cpp on_image_render()  => 每一帧都会传入 on_image_render 函数中,模型推理标注也都在里面

            }
            @Override
            public void surfaceCreated(SurfaceHolder holder) {}
            @Override
            public void surfaceDestroyed(SurfaceHolder holder) {}
        });

    }

    @Override
    public void onResume(){
        super.onResume();
        woker.openCamera(cameraId.get()?1:0);
    }

    @Override
    public void onPause(){
        super.onPause();
        woker.closeCamera();
    }








}
