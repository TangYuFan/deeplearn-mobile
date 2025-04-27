package com.example.android_tflite_lyra_demo;

import android.app.Activity;

import android.content.res.AssetManager;
import android.util.Log;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;
import java.util.BitSet;
import java.util.HashMap;
import java.util.Map;
import org.tensorflow.lite.Interpreter;


/**
 * @desc : google lyra 3.2kbps inference
 * @auth : tyf
 * @date : 2025-04-27 16:05:50
*/
public class Codec {

    private static String TAG = Codec.class.getName();

    private static String codec1ModlePath = null;

    // model path
    private static Interpreter lyragan = null;
    private static Interpreter quantizer = null;
    private static Interpreter soundstream_encoder = null;

    // file init
    public static void initModelFile(Activity context) {
        AssetManager assetManager = context.getAssets();
        String path = context.getFilesDir().getAbsolutePath();

        String assetPath = "codec1_model_coeffs";
        String dstPath = path + "/codec1/";

        codec1ModlePath = dstPath;
        try {
            Log.d(TAG, "App Path：" + dstPath + "，model：" + assetManager.list(assetPath).length);
            File par = new File(dstPath);
            if (!par.exists()) {
                par.mkdir();
            }
            Arrays.stream(assetManager.list(assetPath)).forEach(n -> {
                String fp = dstPath + n;
                File outFile = new File(fp);
                Log.d(TAG, "file：" + fp + "，exist：" + outFile.exists());
                if (!outFile.exists()) {
                    try (InputStream in = assetManager.open(assetPath + "/" + n);
                         FileOutputStream out = new FileOutputStream(outFile)) {
                        byte[] buffer = new byte[1024];
                        int read;
                        while ((read = in.read(buffer)) != -1) {
                            out.write(buffer, 0, read);
                        }
                        out.flush();
                        Log.d(TAG, "File copied to " + fp);
                    } catch (IOException e) {
                        Log.d(TAG, "Failed to copy file " + fp, e);
                    }
                }
            });
        } catch (Exception e) {
            Log.d(TAG, "model init fail：" + e.getMessage() + "," + e.getCause());
        }
    }

    // model init
    public static void initCodec(Activity context) {

        try {

            String model1Path = codec1ModlePath + "lyragan.tflite";
            String model2Path = codec1ModlePath + "quantizer.tflite";
            String model3Path = codec1ModlePath + "soundstream_encoder.tflite";

            Interpreter.Options options = new Interpreter.Options();
            lyragan = new Interpreter(loadModelFile(context, model1Path), options);
            quantizer = new Interpreter(loadModelFile(context, model2Path), options);
            soundstream_encoder = new Interpreter(loadModelFile(context, model3Path), options);

            logModelInfo(lyragan, "lyragan");
            logModelInfo(quantizer, "quantizer");
            logModelInfo(soundstream_encoder, "soundstream_encoder");

        } catch (Exception e) {
            e.printStackTrace();
            Log.e(TAG, "model init fail: " + e.getMessage());
        }
    }

    // show model input/output shape
    private static void logModelInfo(Interpreter interpreter, String modelName) {

        Log.d(TAG, "-------------------");
        int inputCount = interpreter.getInputTensorCount();
        int outputCount = interpreter.getOutputTensorCount();

        Log.d(TAG, "ModelName：" + modelName);
        for (int i = 0; i < inputCount; i++) {
            Log.d(TAG, " Input " + i + " => " + Arrays.toString(interpreter.getInputTensor(i).shape()));
        }
        for (int i = 0; i < outputCount; i++) {
            Log.d(TAG, " Output " + i + " => " + Arrays.toString(interpreter.getOutputTensor(i).shape()));
        }

        String[] signatureKeys = interpreter.getSignatureKeys();
        for (String signatureKey : signatureKeys) {
            Log.d(TAG, "Signature: " + signatureKey);
            String[] signatureInputs = interpreter.getSignatureInputs(signatureKey);
            for (String inputName : signatureInputs) {
                Log.d(TAG, "  Signature Input: " + inputName + " => "
                        + Arrays.toString(interpreter.getInputTensorFromSignature(inputName, signatureKey).shape()));
            }

            String[] signatureOutputs = interpreter.getSignatureOutputs(signatureKey);
            for (String outputName : signatureOutputs) {
                Log.d(TAG, "  Signature Output: " + outputName + " => "
                        + Arrays.toString(interpreter.getOutputTensorFromSignature(outputName, signatureKey).shape()));
            }
        }
        Log.d(TAG, "-------------------");
    }


    // get model file
    private static MappedByteBuffer loadModelFile(Activity context, String modelPath) throws IOException {
        FileInputStream inputStream = new FileInputStream(modelPath);
        FileChannel fileChannel = inputStream.getChannel();
        long fileSize = fileChannel.size();
        Log.d(TAG, "Model Size：" + fileSize);
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, 0, fileSize);
    }


    // 特征提取 float32[1,320] => float32[1,1,64]
    private static float[] extractFeatures(short[] frame) {
        float[] input = new float[frame.length];
        for (int i = 0; i < frame.length; i++) {
            input[i] = (float) frame[i] / (float) Short.MIN_VALUE;
        }
        float[][][] output = new float[1][1][64];
        soundstream_encoder.run(input, output);
        return output[0][0];
    }


    // 特征量化 encode：float32[1,1,64] + encode_num_quantizers => int32[46,1,1] => 64位二进制（8字节）
    private static String quantize(float[] features) {
        String signatureKey = "encode";
        Map<String, Object> inputs = new HashMap<>();
        Map<String, Object> outputs = new HashMap<>();
        float[][][] input_frames = new float[1][1][64];
        input_frames[0][0] = features;
        inputs.put("input_frames", input_frames);
        int bitsPerQuantizer = 4;
        int requiredQuantizers = 64 / bitsPerQuantizer;
        int[] numQuantizers = new int[1];
        numQuantizers[0] = requiredQuantizers;
        inputs.put("num_quantizers", numQuantizers);
        outputs.put("output_0", new int[46][1][1]);
        quantizer.runSignature(inputs, outputs, signatureKey);
        int[][][] quantizedData = (int[][][]) outputs.get("output_0");
        BitSet bitSet = new BitSet(64);
        for (int i = 0; i < requiredQuantizers; i++) {
            int value = quantizedData[i][0][0];
            int shift = (requiredQuantizers - i - 1) * bitsPerQuantizer;
            for (int bit = 0; bit < bitsPerQuantizer; bit++) {
                if (((value >> bit) & 1) != 0) {
                    bitSet.set(shift + bit);
                }
            }
        }
        StringBuilder sb = new StringBuilder();
        for (int i = 63; i >= 0; i--) {
            sb.append(bitSet.get(i) ? '1' : '0');
        }
        return sb.toString();
    }

    // 特征解量化 decode：64位二进制（8字节）=> int32[46,1,1] => float32[1,1,64]
    private static float[] dequantize(String quantizedFeatures) {
        String signatureKey = "decode";
        final int numBits = 64;
        final int bitsPerQuantizer = 4;  // 对应不同级别码率
        final int requiredQuantizers = numBits / bitsPerQuantizer;
        final int maxNumQuantizers = 184 / bitsPerQuantizer;
        int[] quantizedData = new int[maxNumQuantizers];
        for (int i = 0; i < requiredQuantizers; i++) {
            int startBit = i * bitsPerQuantizer;
            int endBit = startBit + bitsPerQuantizer;
            String bits = quantizedFeatures.substring(startBit, endBit);
            quantizedData[i] = Integer.parseInt(bits, 2);
        }
        for (int i = requiredQuantizers; i < maxNumQuantizers; i++) {
            quantizedData[i] = -1;
        }
        Map<String, Object> inputs = new HashMap<>();
        Map<String, Object> outputs = new HashMap<>();
        inputs.put("encoding_indices", quantizedData);
        outputs.put("output_0", new float[64]);
        quantizer.runSignature(inputs, outputs, signatureKey);
        float[] decodedFeatures = (float[]) outputs.get("output_0");
        return decodedFeatures;
    }

    // 特征还原 float32[1,1,64] => float32[1,320]
    private static short[] restoreFrame(float[] features) {
        float[][][] input = new float[1][1][64];
        input[0][0] = features;
        float[][] output = new float[1][320];
        // 推理获取输出
        lyragan.run(input, output);
        short[] frame = new short[320];
        // 每个值是 float64，也就是 -1 ~ 1 之前，转为 short 音频幅值
        for (int i = 0; i < output[0].length; i++) {
            float value = output[0][i];
            frame[i] = (short) (value * Short.MAX_VALUE);
        }
        return frame;
    }



    public static byte[] encode(byte[] data) {
        // 16 位深，两个字节为一个样本
        short[] frame = new short[data.length / 2];
        boolean isLittleEndian = ByteOrder.nativeOrder() == ByteOrder.LITTLE_ENDIAN;

        for (int i = 0; i < frame.length; i++) {
            if (isLittleEndian) {
                frame[i] = (short) ((data[2 * i] & 0xFF) | (data[2 * i + 1] << 8)); // 小端
            } else {
                frame[i] = (short) (((data[2 * i + 1] & 0xFF) << 8) | (data[2 * i] & 0xFF)); // 大端
            }
        }

        float[] features = extractFeatures(frame);
        String quantizedFeatures = quantize(features);
        return quantizedFeatures.getBytes();
    }

    public static byte[] decode(byte[] data) {

        String quantizedFeatures = new String(data);
        float[] decodedFeatures = dequantize(quantizedFeatures);
        short[] restoredFrame = restoreFrame(decodedFeatures);
        byte[] outputData = new byte[restoredFrame.length * 2];

        boolean isLittleEndian = ByteOrder.nativeOrder() == ByteOrder.LITTLE_ENDIAN;

        for (int i = 0; i < restoredFrame.length; i++) {
            if (isLittleEndian) {
                outputData[2 * i] = (byte) (restoredFrame[i] & 0xFF);
                outputData[2 * i + 1] = (byte) ((restoredFrame[i] >> 8) & 0xFF); // 小端
            } else {
                outputData[2 * i] = (byte) ((restoredFrame[i] >> 8) & 0xFF); // 大端
                outputData[2 * i + 1] = (byte) (restoredFrame[i] & 0xFF);
            }
        }
        return outputData;
    }



}
