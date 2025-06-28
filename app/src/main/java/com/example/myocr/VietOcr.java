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
            float[][][] logits = (float[][][]) result.get(0).getValue();
            return vocab.decode(logits);
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
}