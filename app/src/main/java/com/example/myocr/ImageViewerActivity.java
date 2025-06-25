package com.example.myocr;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.graphics.PointF;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.View;
import android.view.ViewTreeObserver;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import androidx.exifinterface.media.ExifInterface;

import com.yalantis.ucrop.UCrop;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import org.opencv.core.CvType;
import org.opencv.core.Core;
import org.opencv.core.Scalar;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.stream.Collectors;

import org.opencv.core.MatOfDouble;

public class ImageViewerActivity extends AppCompatActivity implements ThumbnailAdapter.OnThumbnailClickListener {

    private enum State {
        VIEWING, CROPPED
    }

    private State currentState = State.VIEWING;
    
    private ArrayList<Uri> sourceUris;
    private Bitmap currentOriginalBitmap;
    private PointF[][] savedCorners;
    private List<Bitmap> previewBitmaps = new ArrayList<>();
    private List<Bitmap> thumbnailBitmaps = new ArrayList<>();

    private int currentPosition = 0;

    private ImageView fullScreenImageView;
    private PolygonView polygonView;
    private RecyclerView thumbnailRecyclerView;
    private ThumbnailAdapter thumbnailAdapter;
    private Button btnEdit, btnConfirm;
    private ProgressBar progressBar;
    private TextView tvStatusIndicator;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_image_viewer);
        initViews();
        
        sourceUris = getIntent().getParcelableArrayListExtra("image_uris");
        if (sourceUris == null || sourceUris.isEmpty()) {
            Toast.makeText(this, "No images to process.", Toast.LENGTH_SHORT).show();
            finish();
            return;
        }

        setupRecyclerView();
        setupClickListeners();
        loadImages(sourceUris);
    }

    private void initViews() {
        fullScreenImageView = findViewById(R.id.fullScreenImageView);
        btnEdit = findViewById(R.id.btnEdit);
        btnConfirm = findViewById(R.id.btnConfirm);
        polygonView = findViewById(R.id.polygonView);
        progressBar = findViewById(R.id.image_viewer_progress);
        thumbnailRecyclerView = findViewById(R.id.thumbnailRecyclerView);
        tvStatusIndicator = findViewById(R.id.tvStatusIndicator);
    }

    private void setupRecyclerView() {
        thumbnailAdapter = new ThumbnailAdapter(this, previewBitmaps, this::displayImageAtPosition);
        thumbnailRecyclerView.setLayoutManager(new LinearLayoutManager(this, LinearLayoutManager.HORIZONTAL, false));
        thumbnailRecyclerView.setAdapter(thumbnailAdapter);
    }

    private void setupClickListeners() {
        btnEdit.setOnClickListener(v -> {
            if (currentState == State.CROPPED) resetToPreviewState();
            else startImageEditor();
        });

        btnConfirm.setOnClickListener(v -> {
            if (currentState == State.CROPPED) {
                saveAndFinish();
            } else {
                applyCropAndShowPreview();
            }
        });
    }

    private void loadImages(ArrayList<Uri> uris) {
        progressBar.setVisibility(View.VISIBLE);
        new Thread(() -> {
            // Clear all previous data
            previewBitmaps.clear();
            thumbnailBitmaps.clear();
            savedCorners = new PointF[uris.size()][];

            // Only load lightweight previews and thumbnails
            for (Uri uri : uris) {
                try {
                    // Load the full bitmap temporarily to create derivatives
                    Bitmap originalBitmap = getCorrectlyOrientedBitmap(uri);
                    
                    previewBitmaps.add(createPreviewBitmap(originalBitmap));
                    thumbnailBitmaps.add(Bitmap.createScaledBitmap(originalBitmap, 150, 150, true));

                    // IMPORTANT: Release the full-size bitmap immediately
                    originalBitmap.recycle(); 
                    
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }

            runOnUiThread(() -> {
                progressBar.setVisibility(View.GONE);
                if (!previewBitmaps.isEmpty()) {
                    thumbnailAdapter.notifyDataSetChanged();
                    displayImageAtPosition(0);
                } else {
                    Toast.makeText(this, "Failed to load any images.", Toast.LENGTH_SHORT).show();
                    finish();
                }
            });
        }).start();
    }
    
    private void displayImageAtPosition(int position) {
        if (position < 0 || position >= previewBitmaps.size()) return;
        
        currentPosition = position;
        thumbnailAdapter.setSelectedPosition(position);
        
        // 1. Immediately display the cached preview and reset UI
        fullScreenImageView.setImageBitmap(previewBitmaps.get(position));
        resetToPreviewState();
        polygonView.setDefaultCorners(fullScreenImageView.getWidth(), fullScreenImageView.getHeight());

        // 2. Load the full bitmap and find corners in the background
        progressBar.setVisibility(View.VISIBLE);
        tvStatusIndicator.setVisibility(View.VISIBLE);
        new Thread(() -> {
            // Recycle the previous original bitmap if it exists
            if (currentOriginalBitmap != null && !currentOriginalBitmap.isRecycled()) {
                currentOriginalBitmap.recycle();
            }
            
            try {
                // Load the new full-resolution original bitmap
                currentOriginalBitmap = getCorrectlyOrientedBitmap(sourceUris.get(position));

                PointF[] cornersToDisplay = null;
                // Priority 1: Use user-saved corners if they exist
                if (savedCorners[position] != null) {
                    cornersToDisplay = savedCorners[position];
                } else {
                // Priority 2: Try to auto-detect corners
                    PointF[] detectedCorners = findDocumentCorners(currentOriginalBitmap);
                    if (detectedCorners != null) {
                        cornersToDisplay = orderPoints(detectedCorners);
                    }
                }
                
                final PointF[] finalCorners = cornersToDisplay;

                runOnUiThread(() -> {
                    // Check if we are still on the same image
                    if (currentPosition == position) {
                         if (finalCorners != null) {
                            PointF[] viewCorners = transformCorners(finalCorners, fullScreenImageView);
                            polygonView.setPoints(viewCorners);
                        } else {
                            polygonView.setDefaultCorners(fullScreenImageView.getWidth(), fullScreenImageView.getHeight());
                        }
                    }
                    progressBar.setVisibility(View.GONE);
                    tvStatusIndicator.setVisibility(View.GONE);
                });

            } catch (IOException e) {
                e.printStackTrace();
                runOnUiThread(() -> {
                    progressBar.setVisibility(View.GONE);
                    tvStatusIndicator.setVisibility(View.GONE);
                });
            }
        }).start();
    }

    @Override
    public void onThumbnailClick(int position) {
        displayImageAtPosition(position);
    }

    private void applyCropAndShowPreview() {
        PointF[] viewCorners = polygonView.getPoints();
        PointF[] bitmapCorners = mapViewPointsToBitmapPoints(viewCorners, fullScreenImageView);
        if (bitmapCorners == null || currentOriginalBitmap == null) return;
        
        savedCorners[currentPosition] = orderPoints(bitmapCorners);
        
        Bitmap croppedColor = performPerspectiveTransform(currentOriginalBitmap, savedCorners[currentPosition]);
        
        if (croppedColor != null) {
            Bitmap finalProcessed = createFinalProcessedBitmap(croppedColor);
            fullScreenImageView.setImageBitmap(finalProcessed);
            croppedColor.recycle(); // We don't need the intermediate color crop anymore
        }
        
        enterCroppedState();
    }
    
    private void saveAndFinish() {
        ArrayList<Uri> finalUris = new ArrayList<>();
        ExecutorService executor = Executors.newSingleThreadExecutor();
        Handler handler = new Handler(Looper.getMainLooper());
        progressBar.setVisibility(View.VISIBLE);

        executor.execute(() -> {
            for (int i = 0; i < sourceUris.size(); i++) {
                Bitmap bitmapToProcess = null;
                Bitmap finalBitmapToSave = null;
                Uri uri = null;
                try {
                    bitmapToProcess = getCorrectlyOrientedBitmap(sourceUris.get(i));
                    
                    if (savedCorners[i] != null) {
                        Bitmap croppedBitmap = performPerspectiveTransform(bitmapToProcess, savedCorners[i]);
                        finalBitmapToSave = createFinalProcessedBitmap(croppedBitmap);
                        croppedBitmap.recycle();
                    } else {
                        finalBitmapToSave = createFinalProcessedBitmap(bitmapToProcess);
                    }
                    
                    if (finalBitmapToSave != null) {
                        uri = saveBitmapToFile(finalBitmapToSave);
                        finalBitmapToSave.recycle();
                    }
                    
                    if (uri != null) {
                        finalUris.add(uri);
                    }
                } catch (IOException e) {
                    e.printStackTrace();
                } finally {
                    if (bitmapToProcess != null && !bitmapToProcess.isRecycled()) {
                        bitmapToProcess.recycle(); // release memory for the original
                    }
                }
            }

            handler.post(() -> {
                progressBar.setVisibility(View.GONE);
                Intent resultIntent = new Intent();
                resultIntent.putParcelableArrayListExtra("processed_uris", finalUris);
                setResult(RESULT_OK, resultIntent);
                finish();
            });
        });
    }

    private void enterCroppedState() {
        currentState = State.CROPPED;
        polygonView.setVisibility(View.GONE);
        btnConfirm.setText(R.string.save);
        btnEdit.setText(R.string.reset);
    }
    
    private void resetToPreviewState() {
        currentState = State.VIEWING;
        // Use the cached preview bitmap, don't regenerate it.
        if(currentPosition < previewBitmaps.size()){
            fullScreenImageView.setImageBitmap(previewBitmaps.get(currentPosition));
        }
        polygonView.setVisibility(View.VISIBLE);
        // We now need to re-fetch corners if user resets
        if(savedCorners != null && currentPosition < savedCorners.length) {
            // When resetting, we clear the saved state for this image
            // so that it can be auto-detected again if needed.
             savedCorners[currentPosition] = null;
        }

        btnConfirm.setText(R.string.confirm);
        btnEdit.setText(R.string.edit_image);
    }

    private void startImageEditor() {
        if (currentOriginalBitmap == null || currentOriginalBitmap.isRecycled()) {
             Toast.makeText(this, "Please wait for the image to load.", Toast.LENGTH_SHORT).show();
            return;
        }

        // Save the current original bitmap to a temporary file to send to the editor
        File tempFile = new File(getCacheDir(), "temp_for_edit.jpg");
        try (FileOutputStream out = new FileOutputStream(tempFile)) {
            currentOriginalBitmap.compress(Bitmap.CompressFormat.JPEG, 90, out);
            Uri uriToSend = Uri.fromFile(tempFile);
            
            File destinationFile = new File(getCacheDir(), "edited_image.jpg");
            Uri destinationUri = Uri.fromFile(destinationFile);

            UCrop.of(uriToSend, destinationUri)
                .withOptions(new UCrop.Options())
                .start(this);

        } catch (IOException e) {
            e.printStackTrace();
            Toast.makeText(this, "Error preparing image for editor.", Toast.LENGTH_SHORT).show();
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK && requestCode == UCrop.REQUEST_CROP) {
                final Uri resultUri = UCrop.getOutput(data);
                if (resultUri != null) {
                try (InputStream inputStream = getContentResolver().openInputStream(resultUri)) {
                    Bitmap editedBitmap = BitmapFactory.decodeStream(inputStream);
                    
                    // Replace the current original bitmap with the edited one
                    if (currentOriginalBitmap != null && !currentOriginalBitmap.isRecycled()) {
                        currentOriginalBitmap.recycle();
                    }
                    currentOriginalBitmap = editedBitmap;
                    
                    // Also update the preview
                    previewBitmaps.set(currentPosition, createPreviewBitmap(editedBitmap));

                    // Refresh the display with the new bitmap
                    displayImageAtPosition(currentPosition);

                } catch (IOException e) {
                    e.printStackTrace();
                    Toast.makeText(this, "Failed to load edited image.", Toast.LENGTH_SHORT).show();
                }
            }
        } else if (resultCode == UCrop.RESULT_ERROR) {
            final Throwable cropError = UCrop.getError(data);
            Toast.makeText(this, "Crop error: " + cropError.getMessage(), Toast.LENGTH_SHORT).show();
        }
    }

    private Bitmap performPerspectiveTransform(Bitmap bitmap, PointF[] corners) {
        if (bitmap == null || bitmap.isRecycled()) {
            return null;
        }
        
        Mat srcMat = new Mat();
        Utils.bitmapToMat(bitmap, srcMat);

        MatOfPoint2f srcQuad = new MatOfPoint2f(
            new Point(corners[0].x, corners[0].y),
            new Point(corners[1].x, corners[1].y),
            new Point(corners[2].x, corners[2].y),
            new Point(corners[3].x, corners[3].y)
        );

        double widthA = Math.sqrt(Math.pow(corners[2].x - corners[3].x, 2) + Math.pow(corners[2].y - corners[3].y, 2));
        double widthB = Math.sqrt(Math.pow(corners[1].x - corners[0].x, 2) + Math.pow(corners[1].y - corners[0].y, 2));
        int width = (int) Math.max(widthA, widthB);

        double heightA = Math.sqrt(Math.pow(corners[1].x - corners[2].x, 2) + Math.pow(corners[1].y - corners[2].y, 2));
        double heightB = Math.sqrt(Math.pow(corners[0].x - corners[3].x, 2) + Math.pow(corners[0].y - corners[3].y, 2));
        int height = (int) Math.max(heightA, heightB);

        MatOfPoint2f dstQuad = new MatOfPoint2f(
            new Point(0, 0),
            new Point(width - 1, 0),
            new Point(width - 1, height - 1),
            new Point(0, height - 1)
        );

        Mat perspectiveTransform = Imgproc.getPerspectiveTransform(srcQuad, dstQuad);
        Mat scanned = new Mat();
        Imgproc.warpPerspective(srcMat, scanned, perspectiveTransform, new Size(width, height));

        Bitmap result = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(scanned, result);
        return result;
    }

    private Bitmap createPreviewBitmap(Bitmap bitmap) {
        Mat srcMat = new Mat();
        Utils.bitmapToMat(bitmap, srcMat);
        Mat grayMat = new Mat();
        Imgproc.cvtColor(srcMat, grayMat, Imgproc.COLOR_RGB2GRAY);
        Mat blurredMat = new Mat();
        Imgproc.GaussianBlur(grayMat, blurredMat, new Size(5, 5), 0);
        Mat threshMat = new Mat();
        Imgproc.adaptiveThreshold(blurredMat, threshMat, 255, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY, 21, 5);
        Bitmap preview = Bitmap.createBitmap(threshMat.cols(), threshMat.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(threshMat, preview);
        srcMat.release();
        grayMat.release();
        blurredMat.release();
        threshMat.release();
        return preview;
    }

    private PointF[] findDocumentCorners(Bitmap bitmap) {
        Mat srcMat = new Mat();
        Utils.bitmapToMat(bitmap, srcMat);

        Mat grayMat = new Mat();
        Imgproc.cvtColor(srcMat, grayMat, Imgproc.COLOR_RGB2GRAY);

        Mat blurredMat = new Mat();
        Imgproc.GaussianBlur(grayMat, blurredMat, new Size(5, 5), 0);

        // Automatically determine Canny thresholds based on image's mean intensity.
        // This is more robust than fixed thresholds for varying lighting conditions.
        MatOfDouble mean = new MatOfDouble();
        MatOfDouble stddev = new MatOfDouble();
        Core.meanStdDev(grayMat, mean, stddev);
        double meanVal = mean.get(0,0)[0];
        double sigma = 0.33;
        double lowerThreshold = Math.max(0, (1.0 - sigma) * meanVal);
        double upperThreshold = Math.min(255, (1.0 + sigma) * meanVal);
        mean.release();
        stddev.release();
        
        // Canny edge detection is generally more robust for finding document edges
        Mat cannyMat = new Mat();
        Imgproc.Canny(blurredMat, cannyMat, lowerThreshold, upperThreshold);

        // Dilate the edges to help close gaps in the detected contours
        Mat dilatedMat = new Mat();
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5, 5));
        Imgproc.dilate(cannyMat, dilatedMat, kernel);

        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(dilatedMat, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        if (contours.isEmpty()) {
            // Release memory
            srcMat.release();
            grayMat.release();
            blurredMat.release();
            cannyMat.release();
            dilatedMat.release();
            kernel.release();
            hierarchy.release();
            return null;
        }

        Collections.sort(contours, (c1, c2) -> Double.compare(Imgproc.contourArea(c2), Imgproc.contourArea(c1)));

        for (MatOfPoint contour : contours) {
            MatOfPoint2f contour2f = new MatOfPoint2f(contour.toArray());
            double peri = Imgproc.arcLength(contour2f, true);
            MatOfPoint2f approx = new MatOfPoint2f();
            Imgproc.approxPolyDP(contour2f, approx, 0.02 * peri, true);

            if (approx.rows() == 4) {
                double contourArea = Imgproc.contourArea(approx);
                double imgArea = srcMat.rows() * srcMat.cols();
                
                if (contourArea / imgArea > 0.1 && contourArea / imgArea < 0.95 && Imgproc.isContourConvex(new MatOfPoint(approx.toArray()))) {
                    
                    Mat mask = Mat.zeros(srcMat.size(), CvType.CV_8UC1);
                    Imgproc.drawContours(mask, Arrays.asList(new MatOfPoint(approx.toArray())), -1, new Scalar(255), -1);
                    
                    Scalar meanScalar = Core.mean(srcMat, mask);
                    double meanBrightness = (meanScalar.val[0] + meanScalar.val[1] + meanScalar.val[2]) / 3.0;

                    mask.release();

                    if (meanBrightness > 110) {
                        Point[] foundPoints = approx.toArray();
                        PointF[] result = new PointF[4];
                        for (int i = 0; i < 4; i++) {
                            result[i] = new PointF((float)foundPoints[i].x, (float)foundPoints[i].y);
                        }
                        approx.release();
                        contour2f.release();
                        
                        // Release all Mats
                        srcMat.release();
                        grayMat.release();
                        blurredMat.release();
                        cannyMat.release();
                        dilatedMat.release();
                        kernel.release();
                        hierarchy.release();
                        return result;
                    }
                }
            }
            approx.release();
            contour2f.release();
        }
        srcMat.release();
        grayMat.release();
        blurredMat.release();
        cannyMat.release();
        dilatedMat.release();
        kernel.release();
        hierarchy.release();
        return null;
    }

    private PointF[] orderPoints(PointF[] points) {
        List<PointF> pointList = new ArrayList<>(Arrays.asList(points));
        
        pointList.sort(Comparator.comparing(p -> p.x + p.y));
        
        PointF tl = pointList.get(0);
        PointF br = pointList.get(3);
        
        pointList.sort(Comparator.comparing(p -> p.y - p.x));
        
        PointF tr = pointList.get(0);
        PointF bl = pointList.get(3);

        return new PointF[]{tl, tr, br, bl};
    }
    
    private PointF[] transformCorners(PointF[] srcCorners, ImageView imageView) {
        PointF[] dstCorners = new PointF[4];
        Matrix matrix = new Matrix(imageView.getImageMatrix());

        float[] pts = new float[8];
        for (int i = 0; i < 4; i++) {
            pts[i*2] = srcCorners[i].x;
            pts[i*2 + 1] = srcCorners[i].y;
        }
        matrix.mapPoints(pts);
        for (int i = 0; i < 4; i++) {
            dstCorners[i] = new PointF(pts[i*2], pts[i*2 + 1]);
        }
        return dstCorners;
    }

    private PointF[] mapViewPointsToBitmapPoints(PointF[] viewPoints, ImageView imageView) {
        Matrix inverseMatrix = new Matrix();
        if (!imageView.getImageMatrix().invert(inverseMatrix)) {
            // Matrix is not invertible, cannot proceed.
            return null;
        }

        PointF[] bitmapPoints = new PointF[4];
        float[] pts = new float[8];
        for (int i = 0; i < 4; i++) {
            pts[i*2] = viewPoints[i].x;
            pts[i*2 + 1] = viewPoints[i].y;
        }

        inverseMatrix.mapPoints(pts);

        for (int i = 0; i < 4; i++) {
            bitmapPoints[i] = new PointF(pts[i*2], pts[i*2 + 1]);
        }
        return bitmapPoints;
    }

    private Uri saveBitmapToFile(Bitmap bitmap) {
        File outputDir = new File(getCacheDir(), "processed_images");
        if (!outputDir.exists()) {
            outputDir.mkdirs();
        }
        File outputFile = new File(outputDir, "processed_" + System.currentTimeMillis() + ".jpg");
        try (FileOutputStream out = new FileOutputStream(outputFile)) {
            bitmap.compress(Bitmap.CompressFormat.JPEG, 90, out);
            return Uri.fromFile(outputFile);
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }

    private Bitmap getCorrectlyOrientedBitmap(Uri imageUri) throws IOException {
        InputStream inputStream = getContentResolver().openInputStream(imageUri);
        Bitmap bitmap = BitmapFactory.decodeStream(inputStream);
        inputStream.close();

        // Re-open the input stream to read EXIF data
        inputStream = getContentResolver().openInputStream(imageUri);
        ExifInterface exifInterface = new ExifInterface(inputStream);
        int orientation = exifInterface.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL);

        Matrix matrix = new Matrix();
        switch (orientation) {
            case ExifInterface.ORIENTATION_ROTATE_90:
                matrix.postRotate(90);
                break;
            case ExifInterface.ORIENTATION_ROTATE_180:
                matrix.postRotate(180);
                break;
            case ExifInterface.ORIENTATION_ROTATE_270:
                matrix.postRotate(270);
                break;
        }

        Bitmap rotatedBitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
        if (rotatedBitmap != bitmap) {
            bitmap.recycle();
        }
        inputStream.close();
        return rotatedBitmap;
    }

    private Bitmap createFinalProcessedBitmap(Bitmap colorBitmap) {
        // This function chains all preprocessing steps as requested.
        Mat colorMat = new Mat();
        Utils.bitmapToMat(colorBitmap, colorMat);

        // 1. Grayscale
        Mat grayMat = new Mat();
        Imgproc.cvtColor(colorMat, grayMat, Imgproc.COLOR_RGB2GRAY);
        colorMat.release();

        // 2. Binarize (Adaptive Threshold)
        Mat threshMat = new Mat();
        Imgproc.adaptiveThreshold(grayMat, threshMat, 255,
                Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,
                Imgproc.THRESH_BINARY, 21, 5);
        grayMat.release();

        // 3. Remove Horizontal Lines
        Mat noLinesMat = removeHorizontalLines(threshMat);
        threshMat.release();

        Bitmap finalBitmap = Bitmap.createBitmap(noLinesMat.cols(), noLinesMat.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(noLinesMat, finalBitmap);
        noLinesMat.release();

        return finalBitmap;
    }

    private Mat removeHorizontalLines(Mat binaryMat) {
        // This is a Java/OpenCV implementation of the Python `remove_horizontal_lines`
        Mat inverted = new Mat();
        Core.bitwise_not(binaryMat, inverted);

        Mat horizontalKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(40, 1));
        Mat detectedLines = new Mat();
        Imgproc.morphologyEx(inverted, detectedLines, Imgproc.MORPH_OPEN, horizontalKernel, new Point(-1,-1), 2);

        Core.subtract(inverted, detectedLines, inverted);

        Mat result = new Mat();
        Core.bitwise_not(inverted, result);

        // Release intermediate mats
        inverted.release();
        horizontalKernel.release();
        detectedLines.release();

        return result;
    }
} 