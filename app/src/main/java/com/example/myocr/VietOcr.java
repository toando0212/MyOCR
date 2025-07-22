package com.example.myocr;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.Rect;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import android.util.Log;
import java.util.Arrays;

public class VietOcr {
    // --- Vocabulary and Constants ---
    private static final String VOCAB = "aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!\\\"#$%&''()*+,-./:;<=>?@[\\\\]^_`{|}~";
    private final Vocab vocab;
    private final OrtEnvironment env;
    private final OrtSession crnnSession;

    public VietOcr(Context context) throws Exception {
        vocab = new Vocab(VOCAB);
        env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        crnnSession = env.createSession(assetFilePath(context, "best_checkpoint_printed3.onnx"), opts);
    }

    public void close() throws OrtException {
        crnnSession.close();
    }

    public String predict(Bitmap bitmap) throws Exception {
        // Preprocess image to 3x32x128 (NCHW)
        FloatBuffer imageBuffer = preprocess(bitmap);
        long[] shape = {1, 3, 32, 128};
        try (OnnxTensor inputTensor = OnnxTensor.createTensor(env, imageBuffer, shape)) {
            OrtSession.Result result = crnnSession.run(Collections.singletonMap("input", inputTensor));
            Object output = result.get(0).getValue();
            if (output instanceof float[][][]) {
                float[][][] logits = (float[][][]) output;
                return vocab.decode(logits);
            } else if (output instanceof float[][]) {
                float[][] logits2d = (float[][]) output;
                // Add batch dimension: [1, timesteps, num_classes]
                float[][][] logits = new float[1][logits2d.length][logits2d[0].length];
                for (int t = 0; t < logits2d.length; t++) {
                    System.arraycopy(logits2d[t], 0, logits[0][t], 0, logits2d[0].length);
                }
                return vocab.decode(logits);
            } else {
                throw new IllegalStateException("Unexpected ONNX output type: " + output.getClass().getName());
            }
        }
    }

    private FloatBuffer preprocess(Bitmap bitmap) {
        // Resize and pad to 128x32, normalize theo doctr CRNN
        int targetW = 128, targetH = 32;
        Bitmap resized = Bitmap.createScaledBitmap(bitmap, targetW, targetH, true);
        int[] pixels = new int[targetW * targetH];
        resized.getPixels(pixels, 0, targetW, 0, 0, targetW, targetH);
        float[] mean = {0.694f, 0.695f, 0.693f};
        float[] std = {0.299f, 0.296f, 0.301f};
        FloatBuffer buffer = FloatBuffer.allocate(3 * targetH * targetW);
        for (int y = 0; y < targetH; y++) {
            for (int x = 0; x < targetW; x++) {
                int pixel = pixels[y * targetW + x];
                float r = ((Color.red(pixel) / 255.0f) - mean[0]) / std[0];
                float g = ((Color.green(pixel) / 255.0f) - mean[1]) / std[1];
                float b = ((Color.blue(pixel) / 255.0f) - mean[2]) / std[2];
                buffer.put(y * targetW + x, r);
                buffer.put(targetH * targetW + y * targetW + x, g);
                buffer.put(2 * targetH * targetW + y * targetW + x, b);
            }
        }
        buffer.rewind();
        return buffer;
    }

    private static String assetFilePath(Context context, String assetName) throws Exception {
        File file = new File(context.getCacheDir(), assetName);
        try (InputStream is = context.getAssets().open(assetName);
             FileOutputStream os = new FileOutputStream(file)) {
            byte[] buffer = new byte[4 * 1024];
            int read;
            while ((read = is.read(buffer)) != -1) {
                os.write(buffer, 0, read);
            }
            os.flush();
        }
        return file.getAbsolutePath();
    }

    public List<String> predictBoxes(Bitmap fullImage, List<Rect> boxes) throws Exception {
        List<String> results = new ArrayList<>();
        for (Rect box : boxes) {
            int x = Math.max(0, box.left);
            int y = Math.max(0, box.top);
            int width = Math.min(fullImage.getWidth() - x, box.width());
            int height = Math.min(fullImage.getHeight() - y, box.height());
            if (width < 10 || height < 10) {
                results.add("");
                continue;
            }
            Bitmap cropped = Bitmap.createBitmap(fullImage, x, y, width, height);
            String text = predict(cropped);
            results.add(text);
            cropped.recycle();
        }
        return results;
    }

    //hàm chạy mô hìndh nhận diện
    public List<String> predictBatch(List<Bitmap> bitmaps) throws Exception {
        List<String> results = new ArrayList<>();
        if (bitmaps.isEmpty()) return results;

        int batchSize = bitmaps.size();
        int targetW = 128, targetH = 32;
        FloatBuffer batchBuffer = FloatBuffer.allocate(batchSize * 3 * targetH * targetW);
        float[] mean = {0.694f, 0.695f, 0.693f};
        float[] std = {0.299f, 0.296f, 0.301f};

        for (int batchIndex = 0; batchIndex < batchSize; batchIndex++) {
            Bitmap bitmap = bitmaps.get(batchIndex);
            Bitmap resized = Bitmap.createScaledBitmap(bitmap, targetW, targetH, true);
            int[] pixels = new int[targetW * targetH];
            resized.getPixels(pixels, 0, targetW, 0, 0, targetW, targetH);
            for (int y = 0; y < targetH; y++) {
                for (int x = 0; x < targetW; x++) {
                    int pixel = pixels[y * targetW + x];
                    float r = ((Color.red(pixel) / 255.0f) - mean[0]) / std[0];
                    float g = ((Color.green(pixel) / 255.0f) - mean[1]) / std[1];
                    float b = ((Color.blue(pixel) / 255.0f) - mean[2]) / std[2];
                    int imageBaseIndex = batchIndex * 3 * targetH * targetW;
                    int pixelIndex = y * targetW + x;
                    batchBuffer.put(imageBaseIndex + pixelIndex, r);
                    batchBuffer.put(imageBaseIndex + targetH * targetW + pixelIndex, g);
                    batchBuffer.put(imageBaseIndex + 2 * targetH * targetW + pixelIndex, b);
                }
            }
            resized.recycle();
        }
        batchBuffer.rewind();

        long[] shape = {batchSize, 3, targetH, targetW};
        try (OnnxTensor inputTensor = OnnxTensor.createTensor(env, batchBuffer, shape)) {
            OrtSession.Result result = crnnSession.run(Collections.singletonMap("input", inputTensor));
            Object output = result.get(0).getValue();
            if (output instanceof float[][][]) {
                float[][][] logits = (float[][][]) output;
                for (int batchIndex = 0; batchIndex < batchSize; batchIndex++) {
                    float[][][] singleLogit = new float[1][logits[batchIndex].length][logits[batchIndex][0].length];
                    for (int t = 0; t < logits[batchIndex].length; t++) {
                        System.arraycopy(logits[batchIndex][t], 0, singleLogit[0][t], 0, logits[batchIndex][0].length);
                    }
                    results.add(vocab.decode(singleLogit));
                }
            } else if (output instanceof float[][]) {
                float[][] logits2d = (float[][]) output;
                int timesteps = logits2d.length / batchSize;
                for (int batchIndex = 0; batchIndex < batchSize; batchIndex++) {
                    float[][][] singleLogit = new float[1][timesteps][logits2d[0].length];
                    for (int t = 0; t < timesteps; t++) {
                        System.arraycopy(logits2d[batchIndex * timesteps + t], 0, singleLogit[0][t], 0, logits2d[0].length);
                    }
                    results.add(vocab.decode(singleLogit));
                }
            } else {
                throw new IllegalStateException("Unexpected ONNX output type: " + output.getClass().getName());
            }
        }
        return results;
    }

    public List<String> predictBatchEnglish(List<Bitmap> bitmaps, OrtEnvironment env, OrtSession session, Vocab vocab) throws Exception {
        List<String> results = new ArrayList<>();
        if (bitmaps.isEmpty()) return results;

        int batchSize = bitmaps.size();
        int targetW = 128, targetH = 32;
        FloatBuffer batchBuffer = FloatBuffer.allocate(batchSize * 3 * targetH * targetW);
        float[] mean = {0.694f, 0.695f, 0.693f};
        float[] std = {0.299f, 0.296f, 0.301f};

        for (int batchIndex = 0; batchIndex < batchSize; batchIndex++) {
            Bitmap bitmap = bitmaps.get(batchIndex);
            Bitmap resized = Bitmap.createScaledBitmap(bitmap, targetW, targetH, true);
            int[] pixels = new int[targetW * targetH];
            resized.getPixels(pixels, 0, targetW, 0, 0, targetW, targetH);
            for (int y = 0; y < targetH; y++) {
                for (int x = 0; x < targetW; x++) {
                    int pixel = pixels[y * targetW + x];
                    float r = ((Color.red(pixel) / 255.0f) - mean[0]) / std[0];
                    float g = ((Color.green(pixel) / 255.0f) - mean[1]) / std[1];
                    float b = ((Color.blue(pixel) / 255.0f) - mean[2]) / std[2];
                    int imageBaseIndex = batchIndex * 3 * targetH * targetW;
                    int pixelIndex = y * targetW + x;
                    batchBuffer.put(imageBaseIndex + pixelIndex, r);
                    batchBuffer.put(imageBaseIndex + targetH * targetW + pixelIndex, g);
                    batchBuffer.put(imageBaseIndex + 2 * targetH * targetW + pixelIndex, b);
                }
            }
            resized.recycle();
        }
        batchBuffer.rewind();

        long[] shape = {batchSize, 3, targetH, targetW};
        try (OnnxTensor inputTensor = OnnxTensor.createTensor(env, batchBuffer, shape)) {
            OrtSession.Result result = session.run(Collections.singletonMap("input", inputTensor));
            Object output = result.get(0).getValue();
            if (output instanceof float[][][]) {
                float[][][] logits = (float[][][]) output;
                for (int batchIndex = 0; batchIndex < batchSize; batchIndex++) {
                    float[][][] singleLogit = new float[1][logits[batchIndex].length][logits[batchIndex][0].length];
                    for (int t = 0; t < logits[batchIndex].length; t++) {
                        System.arraycopy(logits[batchIndex][t], 0, singleLogit[0][t], 0, logits[batchIndex][0].length);
                    }
                    results.add(vocab.decode(singleLogit));
                }
            } else if (output instanceof float[][]) {
                float[][] logits2d = (float[][]) output;
                int timesteps = logits2d.length / batchSize;
                for (int batchIndex = 0; batchIndex < batchSize; batchIndex++) {
                    float[][][] singleLogit = new float[1][timesteps][logits2d[0].length];
                    for (int t = 0; t < timesteps; t++) {
                        System.arraycopy(logits2d[batchIndex * timesteps + t], 0, singleLogit[0][t], 0, logits2d[0].length);
                    }
                    results.add(vocab.decode(singleLogit));
                }
            } else {
                throw new IllegalStateException("Unexpected ONNX output type: " + output.getClass().getName());
            }
        }
        return results;
    }
}