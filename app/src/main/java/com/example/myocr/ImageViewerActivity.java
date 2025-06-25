package com.example.myocr;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.PointF;
import android.net.Uri;
import android.os.Bundle;
import android.view.ViewTreeObserver;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

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
import android.graphics.Matrix;
import org.opencv.core.CvType;
import org.opencv.core.Core;
import org.opencv.core.Scalar;

public class ImageViewerActivity extends AppCompatActivity {

    private Uri imageUri;
    private ImageView fullScreenImageView;
    private PolygonView polygonView;
    private Bitmap originalBitmap;
    private int imagePosition = -1;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_image_viewer);

        fullScreenImageView = findViewById(R.id.fullScreenImageView);
        Button btnEdit = findViewById(R.id.btnEdit);
        Button btnConfirm = findViewById(R.id.btnConfirm);
        polygonView = findViewById(R.id.polygonView);

        imageUri = getIntent().getData();
        imagePosition = getIntent().getIntExtra("image_position", -1);

        if (imageUri != null) {
            try {
                InputStream inputStream = getContentResolver().openInputStream(imageUri);
                originalBitmap = BitmapFactory.decodeStream(inputStream);
                if (originalBitmap == null) {
                    Toast.makeText(this, "Failed to load image bitmap.", Toast.LENGTH_SHORT).show();
                    finish();
                    return;
                }
                fullScreenImageView.setImageBitmap(originalBitmap);
                
                // Wait for layout to be complete before trying to detect corners
                fullScreenImageView.getViewTreeObserver().addOnGlobalLayoutListener(new ViewTreeObserver.OnGlobalLayoutListener() {
                    @Override
                    public void onGlobalLayout() {
                        fullScreenImageView.getViewTreeObserver().removeOnGlobalLayoutListener(this);

                        PointF[] detectedCorners = findDocumentCorners(originalBitmap);
                        if (detectedCorners != null && detectedCorners.length == 4) {
                            PointF[] orderedCorners = orderPoints(detectedCorners);
                            PointF[] viewCorners = transformCorners(orderedCorners, fullScreenImageView);
                            polygonView.setPoints(viewCorners);
                        } else {
                            // Fallback to default inset if no document is found
                            polygonView.setDefaultCorners(fullScreenImageView.getWidth(), fullScreenImageView.getHeight());
                        }
                    }
                });

            } catch (IOException e) {
                e.printStackTrace();
                Toast.makeText(this, "Failed to load image: " + e.getMessage(), Toast.LENGTH_SHORT).show();
                finish();
                return;
            }
        } else {
            Toast.makeText(this, "No image URI provided", Toast.LENGTH_SHORT).show();
            finish();
            return;
        }


        btnEdit.setOnClickListener(v -> startImageEditor(imageUri));
        btnConfirm.setOnClickListener(v -> {
            if (originalBitmap == null) {
                Toast.makeText(this, "Cannot process a null image.", Toast.LENGTH_SHORT).show();
                return;
            }
            PointF[] viewCorners = polygonView.getPoints();
            PointF[] bitmapCorners = mapViewPointsToBitmapPoints(viewCorners, fullScreenImageView);

            if (bitmapCorners == null) {
                Toast.makeText(this, "Error transforming points. Cannot crop.", Toast.LENGTH_SHORT).show();
                return;
            }

            Bitmap scanned = performPerspectiveTransform(originalBitmap, bitmapCorners);

            Uri resultUri = saveBitmapToFile(scanned);
            if (resultUri == null) {
                Toast.makeText(this, "Failed to save processed image.", Toast.LENGTH_SHORT).show();
                setResult(RESULT_CANCELED);
            } else {
                Intent resultIntent = new Intent();
                resultIntent.setData(resultUri);
                resultIntent.putExtra("image_position", imagePosition);
                setResult(RESULT_OK, resultIntent);
            }
            finish();
        });

        if (getSupportActionBar() != null) {
            getSupportActionBar().setDisplayHomeAsUpEnabled(true);
            getSupportActionBar().setTitle("Image Viewer");
        }
    }

    private void startImageEditor(Uri sourceUri) {
        if (sourceUri == null) {
            Toast.makeText(this, "No image to edit", Toast.LENGTH_SHORT).show();
            return;
        }

        // Create a destination Uri for the edited image
        File editedImageFile = new File(getCacheDir(), "edited_image.jpg");
        Uri destinationUri = Uri.fromFile(editedImageFile);

        UCrop.Options options = new UCrop.Options();
        // You can customize the options here, for example:
        // options.setToolbarTitle("Edit Image");
        // options.setFreeStyleCropEnabled(true);

        UCrop.of(sourceUri, destinationUri)
                .withOptions(options)
                .start(this);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == UCrop.REQUEST_CROP) {
            if (resultCode == RESULT_OK && data != null) {
                final Uri resultUri = UCrop.getOutput(data);
                if (resultUri != null) {
                    // When uCrop finishes, we treat it as a confirmed edit.
                    // Return the result to MainActivity.
                    Intent resultIntent = new Intent();
                    resultIntent.setData(resultUri);
                    resultIntent.putExtra("image_position", imagePosition);
                    setResult(RESULT_OK, resultIntent);
                    finish();
                } else {
                    Toast.makeText(this, "Failed to get cropped/rotated image URI", Toast.LENGTH_SHORT).show();
                }
            } else if (resultCode == UCrop.RESULT_ERROR) {
                final Throwable cropError = UCrop.getError(data);
                Toast.makeText(this, "Crop/Rotate error: " + cropError.getMessage(), Toast.LENGTH_LONG).show();
            }
        }
    }

    @Override
    public boolean onSupportNavigateUp() {
        finish();
        return true;
    }

    private Bitmap performPerspectiveTransform(Bitmap bitmap, PointF[] corners) {
        Mat src = new Mat();
        Utils.bitmapToMat(bitmap, src);

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
        Imgproc.warpPerspective(src, scanned, perspectiveTransform, new Size(width, height));

        Bitmap result = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(scanned, result);
        return result;
    }

    private PointF[] findDocumentCorners(Bitmap bitmap) {
        Mat srcMat = new Mat();
        Utils.bitmapToMat(bitmap, srcMat);

        Mat grayMat = new Mat();
        Imgproc.cvtColor(srcMat, grayMat, Imgproc.COLOR_RGB2GRAY);

        Mat blurredMat = new Mat();
        Imgproc.GaussianBlur(grayMat, blurredMat, new Size(5, 5), 0);

        Mat threshMat = new Mat();
        Imgproc.adaptiveThreshold(blurredMat, threshMat, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY_INV, 11, 2);
        
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(threshMat, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        if (contours.isEmpty()) {
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
                
                // Add more robust checks, similar to the Python implementation
                if (contourArea / imgArea > 0.1 && contourArea / imgArea < 0.95 && Imgproc.isContourConvex(new MatOfPoint(approx.toArray()))) {
                    
                    // Brightness check
                    Mat mask = Mat.zeros(srcMat.size(), CvType.CV_8UC1);
                    Imgproc.drawContours(mask, Arrays.asList(new MatOfPoint(approx.toArray())), -1, new Scalar(255), -1);
                    
                    Scalar meanScalar = Core.mean(srcMat, mask);
                    double meanBrightness = (meanScalar.val[0] + meanScalar.val[1] + meanScalar.val[2]) / 3.0;

                    mask.release();

                    if (meanBrightness > 110) { // Check if the area is bright enough to be a document
                        Point[] foundPoints = approx.toArray();
                        PointF[] result = new PointF[4];
                        for (int i = 0; i < 4; i++) {
                            result[i] = new PointF((float)foundPoints[i].x, (float)foundPoints[i].y);
                        }
                        approx.release();
                        contour2f.release();
                        return result;
                    }
                }
            }
            approx.release();
            contour2f.release();
        }
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
} 