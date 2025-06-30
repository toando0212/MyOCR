package com.example.myocr;

import android.Manifest;
import android.content.ContentValues;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.Configuration;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.MenuItem;
import android.view.View;
import android.widget.Button;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import java.util.Locale;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.ActionBarDrawerToggle;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.drawerlayout.widget.DrawerLayout;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import androidx.appcompat.widget.Toolbar;

import com.google.android.material.floatingactionbutton.FloatingActionButton;
import com.google.android.material.navigation.NavigationView;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.Map;
import java.util.Collections;
import java.util.stream.Collectors;
import java.util.HashMap;

import okhttp3.*;
import org.json.JSONArray;
import org.json.JSONObject;
import org.json.JSONException;
import android.content.SharedPreferences;
import android.widget.LinearLayout;
import android.util.Base64;
import android.util.Log;
import android.os.Environment;

import com.itextpdf.text.Document;
import com.itextpdf.text.Font;
import com.itextpdf.text.Paragraph;
import com.itextpdf.text.pdf.BaseFont;
import com.itextpdf.text.pdf.PdfWriter;
import org.apache.poi.xwpf.usermodel.XWPFDocument;
import org.apache.poi.xwpf.usermodel.XWPFParagraph;
import org.apache.poi.xwpf.usermodel.XWPFRun;

import org.opencv.android.OpenCVLoader;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import android.graphics.Matrix;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import android.content.res.AssetFileDescriptor;
import java.util.Optional;
import android.graphics.RectF;
import java.util.Comparator;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.RotatedRect;
import org.opencv.core.Point;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.Size;
import org.opencv.core.Core;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.CvType;
import org.opencv.core.Scalar;
import org.opencv.core.Rect;

import smile.clustering.HierarchicalClustering;
import smile.clustering.linkage.WardLinkage;

public class MainActivity extends AppCompatActivity implements ImageAdapter.OnImageClickListener, HistoryAdapter.OnHistorySessionInteractionListener {
    private static final int REQUEST_CAMERA_PERMISSION = 100;
    private static final int REQUEST_WRITE_STORAGE_PERMISSION = 102; // For export
    private static final int REQUEST_CODE_CREATE_PDF = 2001;
    private static final int REQUEST_CODE_CREATE_DOCX = 2002;
    private RecyclerView imageRecyclerView;
    private ImageAdapter imageAdapter;
    private List<Uri> imageUris = new ArrayList<>();
    private Button btnRunOcr, btnExport, btnStopOcr;
    private ProgressBar progressBar;
    private FloatingActionButton fab;
    private Uri cameraImageUri;
    private RadioButton radioVietnamese, radioEnglish;
    private RadioGroup languageRadioGroup;
    private TextView tvSelectLanguage;
    private TextView tvDeleteInstruction;

    private RecyclerView ocrResultRecyclerView;
    private OcrResultAdapter ocrResultAdapter;
    private List<OcrResult> ocrResultList = new ArrayList<>();
    private int currentOcrIndex = -1;
    private volatile boolean stopOcrRequested = false;
    private Call currentOcrCall;
    private VietOcr vietOcr;

    private OrtEnvironment ortEnv;
    private OrtSession ortSession;
    private OrtSession detectionSession;
    private OrtSession englishRecognitionSession;

    private Vocab vietnameseVocab;
    private Vocab englishVocab;

    private ActivityResultLauncher<Intent> pickImageLauncher;
    private ActivityResultLauncher<Intent> captureImageLauncher;
    private ActivityResultLauncher<Intent> imageViewerLauncher;

    private DrawerLayout drawerLayout;
    private ActionBarDrawerToggle toggle;

    // Views in Navigation Drawer
    private LinearLayout guestViewNav;
    private Button btnLoginNav, btnNewSession;
    private RecyclerView historyRecyclerViewNav;
    private HistoryAdapter historyAdapterNav;
    private List<HistorySession> historySessionList = new ArrayList<>();

    private boolean isLoggedIn = false;
    private int userId = -1;
    private final OkHttpClient client = new OkHttpClient.Builder()
            .connectTimeout(60, TimeUnit.SECONDS)
            .writeTimeout(60, TimeUnit.SECONDS)
            .readTimeout(60, TimeUnit.SECONDS)
            .build();
    public static final String BASE_URL = "http://192.168.1.229:5000"; // IMPORTANT: Replace '192.168.x.x' with your actual server IP address

    private Uri pendingExportPdfUri = null;

    static {
        if(OpenCVLoader.initDebug()){
            Log.d("MainActivity", "OpenCV is loaded");
        } else {
            Log.e("MainActivity", "OpenCV is not loaded");
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        // Initialize ONNX Runtime but do not load the heavy models yet.
        try {
            ortEnv = OrtEnvironment.getEnvironment();
        } catch (Exception e) {
            Log.e("MainActivity", "Could not initialize OrtEnvironment.", e);
            Toast.makeText(this, "Failed to initialize ONNX Runtime.", Toast.LENGTH_LONG).show();
            // btnRunOcr could be null here, so we find it first.
            // But it's better to disable it after findViewById
        }
            
        // The vocabulary can be initialized here as it's lightweight.
        // --- VERY IMPORTANT ---
        // The CRNN model file you are using (`crnn_mobilenet_v3_large.onnx`) has an output dimension of 127.
        // This means it was trained with a vocabulary of exactly 126 characters (+1 blank token).
        // The vocabulary string below is a standard 95-character English set and DOES NOT MATCH the model.
        // Using this will lead to incorrect results.
        // ACTION REQUIRED: You MUST find and replace the string below with the correct 126-character
        // vocabulary that was used to train your specific .onnx model for it to work correctly.
        String englishChars = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~°£€¥¢฿àâéèêëîïôùûüçÀÂÉÈÊËÎÏÔÙÛÜÇ";
        englishVocab = new Vocab(englishChars);

        // Đọc ngôn ngữ đã lưu (nếu có)
        String lang = getSharedPreferences("settings", MODE_PRIVATE).getString("lang", "en");
        setLocale(lang);
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Check login status
        SharedPreferences prefs = getSharedPreferences("user_prefs", MODE_PRIVATE);
        isLoggedIn = prefs.getBoolean("isLoggedIn", false);
        userId = prefs.getInt("userId", -1);

        Toolbar toolbar = findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);

        imageRecyclerView = findViewById(R.id.imageRecyclerView);
        btnRunOcr = findViewById(R.id.btnRunOcr);
        btnStopOcr = findViewById(R.id.btnStopOcr);
        btnExport = findViewById(R.id.btnExport);
        tvSelectLanguage = findViewById(R.id.tvSelectLanguage);
        radioVietnamese = findViewById(R.id.radioVietnamese);
        radioEnglish = findViewById(R.id.radioEnglish);
        languageRadioGroup = findViewById(R.id.languageRadioGroup);
        progressBar = findViewById(R.id.progressBar);
        fab = findViewById(R.id.fab);
        tvDeleteInstruction = findViewById(R.id.tv_delete_instruction);

        if (ortEnv == null) {
            btnRunOcr.setEnabled(false);
        }

        // Setup for the selected images RecyclerView
        imageAdapter = new ImageAdapter(this, imageUris, this);
        imageRecyclerView.setLayoutManager(new LinearLayoutManager(this, LinearLayoutManager.HORIZONTAL, false));
        imageRecyclerView.setAdapter(imageAdapter);

        // Setup for the OCR results RecyclerView
        ocrResultRecyclerView = findViewById(R.id.ocrResultRecyclerView);
        ocrResultAdapter = new OcrResultAdapter(this, ocrResultList, radioEnglish.isChecked(), client);
        ocrResultRecyclerView.setLayoutManager(new LinearLayoutManager(this));
        ocrResultRecyclerView.setAdapter(ocrResultAdapter);

        pickImageLauncher = registerForActivityResult(
                new ActivityResultContracts.StartActivityForResult(),
                result -> {
                    if (result.getResultCode() == RESULT_OK && result.getData() != null) {
                        Intent data = result.getData();
                        ArrayList<Uri> selectedUris = new ArrayList<>();
                        if (data.getClipData() != null) {
                            int count = data.getClipData().getItemCount();
                            for (int i = 0; i < count; i++) {
                                Uri imageUri = data.getClipData().getItemAt(i).getUri();
                                if (imageUri != null) {
                                    selectedUris.add(imageUri);
                                }
                            }
                        } else if (data.getData() != null) {
                            // Handle single image selection as well
                            Uri imageUri = data.getData();
                            selectedUris.add(imageUri);
                        }
                        
                        if (!selectedUris.isEmpty()) {
                            Intent intent = new Intent(MainActivity.this, ImageViewerActivity.class);
                            intent.putParcelableArrayListExtra("image_uris", selectedUris);
                            imageViewerLauncher.launch(intent);
                        }
                    }
                }
        );

        captureImageLauncher = registerForActivityResult(
                new ActivityResultContracts.StartActivityForResult(),
                result -> {
                    if (result.getResultCode() == RESULT_OK) {
                        if (cameraImageUri != null) {
                            ArrayList<Uri> selectedUris = new ArrayList<>();
                            selectedUris.add(cameraImageUri);
                            
                            Intent intent = new Intent(MainActivity.this, ImageViewerActivity.class);
                            intent.putParcelableArrayListExtra("image_uris", selectedUris);
                            imageViewerLauncher.launch(intent);
                        }
                    }
                }
        );

        imageViewerLauncher = registerForActivityResult(
                new ActivityResultContracts.StartActivityForResult(),
                result -> {
                    if (result.getResultCode() == RESULT_OK && result.getData() != null) {
                        // This is the result from ImageViewerActivity
                        if (result.getData().hasExtra("processed_uris")) {
                            ArrayList<Uri> processedUris = result.getData().getParcelableArrayListExtra("processed_uris");
                            imageUris.addAll(processedUris);
                            imageAdapter.notifyDataSetChanged();
                            updateDeleteInstructionVisibility();
                            // Clear previous OCR results as the image list has changed
                            ocrResultList.clear();
                            ocrResultAdapter.notifyDataSetChanged();
                            btnRunOcr.setEnabled(!imageUris.isEmpty());
                            btnExport.setEnabled(false); // Disable export until new OCR is run
                        }
                    }
                }
        );
        
        btnRunOcr.setOnClickListener(v -> runOcrOnImages());
        btnStopOcr.setOnClickListener(v -> showStopOcrConfirmationDialog());
        btnExport.setOnClickListener(v -> exportRecognizedText());
        fab.setOnClickListener(view -> showImageSourceDialog());

        // Disable Vietnamese option as vocab is missing
        radioVietnamese.setEnabled(true);

        // Đặt trạng thái radio theo ngôn ngữ đã lưu
        if ("vi".equals(lang)) {
            radioVietnamese.setChecked(true);
        } else {
            radioEnglish.setChecked(true);
        }
        languageRadioGroup.setOnCheckedChangeListener((group, checkedId) -> {
            String newLang = checkedId == R.id.radioVietnamese ? "vi" : "en";
            String currentLang = getSharedPreferences("settings", MODE_PRIVATE).getString("lang", "en");
            if (!newLang.equals(currentLang)) {
                getSharedPreferences("settings", MODE_PRIVATE).edit().putString("lang", newLang).apply();
                setLocale(newLang);
                recreate();
            }
            // Cập nhật adapter khi đổi ngôn ngữ
            if (ocrResultAdapter != null) {
                ocrResultAdapter = new OcrResultAdapter(this, ocrResultList, "en".equals(newLang), client);
                ocrResultRecyclerView.setAdapter(ocrResultAdapter);
            }
        });

        // Set progress bar to indeterminate mode
        progressBar.setIndeterminate(false);
        progressBar.setProgress(0);
        progressBar.setVisibility(View.GONE);

        drawerLayout = findViewById(R.id.drawer_layout);
        NavigationView navigationView = findViewById(R.id.nav_view);

        toggle = new ActionBarDrawerToggle(this, drawerLayout, toolbar, R.string.navigation_drawer_open, R.string.navigation_drawer_close);
        drawerLayout.addDrawerListener(toggle);

        getSupportActionBar().setDisplayHomeAsUpEnabled(true);
        getSupportActionBar().setHomeButtonEnabled(true);

        // Setup views in the navigation drawer
        guestViewNav = navigationView.findViewById(R.id.guest_view_nav);
        btnLoginNav = navigationView.findViewById(R.id.btn_login_nav);
        btnNewSession = navigationView.findViewById(R.id.btn_new_session);
        historyRecyclerViewNav = navigationView.findViewById(R.id.history_recycler_view_nav);

        // Setup adapter for history recyclerview
        historyAdapterNav = new HistoryAdapter(this, historySessionList, this);
        historyRecyclerViewNav.setLayoutManager(new LinearLayoutManager(this));
        historyRecyclerViewNav.setAdapter(historyAdapterNav);

        setupNavigationDrawer();
        adjustNavDrawerForStatusBar();

        updateTexts();
    }

    @Override
    protected void onPostCreate(Bundle savedInstanceState) {
        super.onPostCreate(savedInstanceState);
        toggle.syncState();
    }

    @Override
    public void onConfigurationChanged(@NonNull Configuration newConfig) {
        super.onConfigurationChanged(newConfig);
        toggle.onConfigurationChanged(newConfig);
    }

    @Override
    public boolean onOptionsItemSelected(@NonNull MenuItem item) {
        if (toggle.onOptionsItemSelected(item)) {
            return true;
        }
        return super.onOptionsItemSelected(item);
    }

    private void showImageSourceDialog() {
        String[] options = {getString(R.string.take_photo), getString(R.string.upload_from_device)};
        new AlertDialog.Builder(this)
                .setTitle(getString(R.string.choose_image_source))
                .setItems(options, (dialog, which) -> {
                    if (which == 0) {
                        openCamera();
                    } else if (which == 1) {
                        openGallery();
                    }
                })
                .show();
    }

    private void openCamera() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, REQUEST_CAMERA_PERMISSION);
            return;
        }
        Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        ContentValues values = new ContentValues();
        values.put(MediaStore.Images.Media.TITLE, "New Picture");
        values.put(MediaStore.Images.Media.DESCRIPTION, "From Camera");
        cameraImageUri = getContentResolver().insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);
        intent.putExtra(MediaStore.EXTRA_OUTPUT, cameraImageUri);
        captureImageLauncher.launch(intent);
    }

    private void openGallery() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("image/*");
        intent.putExtra(Intent.EXTRA_ALLOW_MULTIPLE, true);
        pickImageLauncher.launch(Intent.createChooser(intent, "Select Image Source"));
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_CAMERA_PERMISSION) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                openCamera();
            } else {
                Toast.makeText(this, "Camera permission is required to use the camera.", Toast.LENGTH_SHORT).show();
            }
        } else if (requestCode == REQUEST_WRITE_STORAGE_PERMISSION) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                // Permission was granted. Show the format selection dialog.
                showExportFormatDialog();
            } else {
                Toast.makeText(this, "Storage permission is required to export the file.", Toast.LENGTH_SHORT).show();
            }
        }
    }

    private void runOcrOnImages() {
        if (imageUris == null || imageUris.isEmpty()) {
            Toast.makeText(this, "Please select images first.", Toast.LENGTH_SHORT).show();
            return;
        }

        // Lazy load the models on first OCR run
        if (detectionSession == null || englishRecognitionSession == null || vietOcr == null) {
             Toast.makeText(this, "Loading models, please wait...", Toast.LENGTH_LONG).show();
             new Thread(() -> {
                 try {
                    // Load English models
                    detectionSession = ortEnv.createSession(assetFilePath("db_mobilenet_v3_large_copy.onnx"), new OrtSession.SessionOptions()); // Changed to db_mobilenet_v3_large.onnx
                    englishRecognitionSession = ortEnv.createSession(assetFilePath("crnn_mobilenet_v3_large.onnx"), new OrtSession.SessionOptions());

                    // Load Vietnamese models by initializing VietOcr class
                    vietOcr = new VietOcr(MainActivity.this);

                    Log.d("OcrDebugging", "All models loaded successfully.");
                    
                    runOnUiThread(() -> {
                         Toast.makeText(MainActivity.this, "Models loaded. Starting OCR.", Toast.LENGTH_SHORT).show();
                         runOcrOnImagesInternal();
                    });

                 } catch (Exception e) {
                     Log.e("MainActivity", "Failed to load ONNX models", e); 
                     runOnUiThread(() -> Toast.makeText(MainActivity.this, "Error loading models: " + e.getMessage(), Toast.LENGTH_LONG).show());
                 }
             }).start();
        } else {
            runOcrOnImagesInternal();
        }
    }

    private void runOcrOnImagesInternal() {
        stopOcrRequested = false;
        ocrResultList.clear();
        for (Uri uri : imageUris) {
            ocrResultList.add(new OcrResult(uri, true));
        }
        ocrResultAdapter = new OcrResultAdapter(this, ocrResultList, radioEnglish.isChecked(), client);
        ocrResultRecyclerView.setAdapter(ocrResultAdapter);
        // Get the language choice from UI thread here, before starting background processing
        final String selectedLang = radioVietnamese.isChecked() ? "vi" : "en";

        currentOcrIndex = 0;
        updateOcrUiState(true);
        processNextImage(selectedLang);
    }
    
    private void processNextImage(final String lang) {
        if (stopOcrRequested) {
             updateOcrUiState(false);
             Toast.makeText(this, "OCR stopped by user.", Toast.LENGTH_SHORT).show();
             return;
        }
        if (currentOcrIndex < imageUris.size()) {
            Uri imageUri = imageUris.get(currentOcrIndex);
            performOcrForImage(imageUri, currentOcrIndex, lang);
        } else {
            // All images processed
            updateOcrUiState(false);
            // Save to history if logged in
            if (isLoggedIn && !ocrResultList.isEmpty()) {
                saveSessionToHistory();
            }
        }
    }
    
    private void performOcrForImage(Uri imageUri, final int position, final String selectedLang) {
        new Thread(() -> {
            Page resultPage = null;
            try {
                Bitmap originalBitmap = BitmapFactory.decodeStream(getContentResolver().openInputStream(imageUri));
                Mat originalMat = new Mat();
                Utils.bitmapToMat(originalBitmap, originalMat);
                removeHorizontalLines(originalMat);
                Bitmap cleanedBitmap = Bitmap.createBitmap(originalMat.cols(), originalMat.rows(), Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(originalMat, cleanedBitmap);

                // Common steps for both languages
                List<RotatedRect> detectionBoxes = runDetection(cleanedBitmap);
                Log.d("OcrDebugging", "Found " + detectionBoxes.size() + " boxes in detection phase for lang: " + selectedLang);
                List<List<WordBox>> lineWordBoxes = resolveLines(detectionBoxes);

                List<Line> resolvedLines = new ArrayList<>();
                Mat originalMatForCropping = new Mat();
                Utils.bitmapToMat(cleanedBitmap, originalMatForCropping);

                for (List<WordBox> wordBoxLine : lineWordBoxes) {
                    List<Word> wordsInLine = new ArrayList<>();
                    for (WordBox wordBox : wordBoxLine) {
                        Rect uprightBox = wordBox.box.boundingRect();
                        int x = Math.max(0, uprightBox.x);
                        int y = Math.max(0, uprightBox.y);
                        int width = Math.min(originalMatForCropping.cols() - x, uprightBox.width);
                        int height = Math.min(originalMatForCropping.rows() - y, uprightBox.height);

                        if (width <= 1 || height <= 1) continue;

                        Rect clampedBox = new Rect(x, y, width, height);
                        Mat croppedMat = new Mat(originalMatForCropping, clampedBox);
                        Bitmap croppedBitmap = Bitmap.createBitmap(croppedMat.cols(), croppedMat.rows(), Bitmap.Config.ARGB_8888);
                        Utils.matToBitmap(croppedMat, croppedBitmap);

                        String recognizedText = "";
                        double confidence = 0.95; // Placeholder confidence

                        if ("vi".equals(selectedLang)) {
                            if (vietOcr == null) throw new IOException("Vietnamese OCR model is not initialized.");
                            recognizedText = vietOcr.predict(croppedBitmap);
                        } else {
                            if (englishRecognitionSession == null || englishVocab == null) throw new IOException("English OCR model is not initialized.");
                            FloatBuffer recInputBuffer = preprocessImageForRecognition(croppedBitmap);
                            long[] recShape = {1, 3, 32, 128};
                            try (OnnxTensor recInputTensor = OnnxTensor.createTensor(ortEnv, recInputBuffer, recShape)) {
                                OrtSession.Result recResult = englishRecognitionSession.run(Collections.singletonMap("input", recInputTensor));
                                float[][][] recOutput = (float[][][]) recResult.get(0).getValue();
                                recognizedText = englishVocab.decode(recOutput);
                            }
                        }

                        wordsInLine.add(new Word(recognizedText, confidence, wordBox.box));
                        croppedMat.release();
                        croppedBitmap.recycle();
                    }

                    if (!wordsInLine.isEmpty()) {
                        RotatedRect lineGeometry = getEnclosingLineBox(wordBoxLine);
                        resolvedLines.add(new Line(wordsInLine, lineGeometry));
                    }
                }
                originalMatForCropping.release();

                // Common steps for both languages
                List<Block> resolvedBlocks = resolveBlocks(resolvedLines);
                Bitmap finalPreviewBitmap = createPreviewWithBlocks(originalBitmap, resolvedBlocks);
                resultPage = new Page(resolvedBlocks, position, originalBitmap.getWidth(), originalBitmap.getHeight(), finalPreviewBitmap);

            } catch (Exception e) {
                Log.e("MainActivity", "Error during OCR for image " + position, e);
                // In case of error, create a placeholder Page object to pass to the UI
                List<Block> errorBlock = new ArrayList<>();
                errorBlock.add(new Block(new ArrayList<>(), new RotatedRect()));
                resultPage = new Page(errorBlock, position, 0, 0, null);
                // We'll handle the text update in updateOcrResult
            }

            // 8. Update UI with the final Page object
            final Page finalResultPage = resultPage;
            final String errorMessage = (resultPage != null && resultPage.getBlocks().isEmpty() || (resultPage != null && resultPage.getBlocks().get(0).getLines().isEmpty()))
                ? "Error during OCR process." : null;
            
            runOnUiThread(() -> {
                if(errorMessage != null){
                     updateOcrResult(position, errorMessage, false, null);
                } else {
                     updateOcrResult(position, finalResultPage, false);
                }
                triggerNextImageProcessing(selectedLang);
            });

        }).start();
    }

    private Bitmap createPreviewWithBlocks(Bitmap sourceBitmap, List<Block> blocks) {
        Bitmap previewBitmap = sourceBitmap.copy(Bitmap.Config.ARGB_8888, true);
        Mat previewMat = new Mat();
        Utils.bitmapToMat(previewBitmap, previewMat);

        for (Block block : blocks) {
            // Draw lines in GREEN
            for(Line line : block.getLines()){
                RotatedRect lineBox = line.getGeometry();
                Point[] lineVertices = new Point[4];
                lineBox.points(lineVertices);
                MatOfPoint linePoints = new MatOfPoint(lineVertices);
                Imgproc.polylines(previewMat, Collections.singletonList(linePoints), true, new Scalar(0, 255, 0), 2); // Green for lines
                linePoints.release();

                // Draw words in BLUE
                for(Word word : line.getWords()){
                    RotatedRect wordBox = word.getGeometry();
                    Point[] wordVertices = new Point[4];
                    wordBox.points(wordVertices);
                    MatOfPoint wordPoints = new MatOfPoint(wordVertices);
                    Imgproc.polylines(previewMat, Collections.singletonList(wordPoints), true, new Scalar(0, 0, 255), 1); // Blue for words
                    wordPoints.release();
                }
            }
        }
        Utils.matToBitmap(previewMat, previewBitmap);
        return previewBitmap;
    }

    private void removeHorizontalLines(Mat image) {
        // This function operates in-place on a color Mat (e.g., RGBA)
        if (image.empty()) return;

        // Work on a copy to avoid modifying the array used in loops
        Mat gray = new Mat();
        Imgproc.cvtColor(image, gray, Imgproc.COLOR_RGBA2GRAY);

        Mat binary = new Mat();
        // Invert the image: Text becomes white, background black. Lines will also be white.
        Imgproc.adaptiveThreshold(gray, binary, 255, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY_INV, 15, -2);

        // Detect horizontal lines
        // The kernel width (e.g., 40) should be adjusted based on the expected line length.
        Mat horizontalKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(40, 1));
        Mat horizontalLines = new Mat();
        Imgproc.morphologyEx(binary, horizontalLines, Imgproc.MORPH_OPEN, horizontalKernel);

        // Dilate the detected lines slightly to ensure they are fully covered during removal
        Imgproc.dilate(horizontalLines, horizontalLines, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(1, 2)));

        // Set the areas of the original image where lines were detected to white
        image.setTo(new Scalar(255, 255, 255, 255), horizontalLines);

        // Release intermediate Mats
        gray.release();
        binary.release();
        horizontalKernel.release();
        horizontalLines.release();
    }

    private List<RotatedRect> runDetection(Bitmap bitmap) throws Exception {
        int targetSize = 1024; // Changed to 1024 to match DBNet input size
        // The ratio of the original image
        double ratio = (double) bitmap.getWidth() / (double) bitmap.getHeight();

        // The size of the resized image
        int resizedWidth, resizedHeight;
        if (ratio > 1) { // Landscape
            resizedWidth = targetSize;
            resizedHeight = (int) (targetSize / ratio);
        } else { // Portrait or Square
            resizedHeight = targetSize;
            resizedWidth = (int) (targetSize * ratio);
        }

        // Prevent dimensions from becoming zero for highly skewed images
        resizedWidth = Math.max(1, resizedWidth);
        resizedHeight = Math.max(1, resizedHeight);

        // Calculate padding
        int top = (targetSize - resizedHeight) / 2;
        int left = (targetSize - resizedWidth) / 2;

        // Updated mean and std for doctr DBNet models
        float[] mean = {0.798f, 0.785f, 0.772f};
        float[] std = {0.264f, 0.2749f, 0.287f};

        FloatBuffer inputBuffer = preprocessImageForDetection(bitmap, targetSize, resizedWidth, resizedHeight, mean, std);
        long[] shape = {1, 3, targetSize, targetSize};

        try (OnnxTensor inputTensor = OnnxTensor.createTensor(ortEnv, inputBuffer, shape)) {
            OrtSession.Result result = detectionSession.run(Collections.singletonMap("input", inputTensor));
            // The model has a single output: logits
            float[][][][] logits = (float[][][][]) result.get(0).getValue();
            // The actual post-processing happens here
            return decodeDetectionOutput(logits, bitmap.getWidth(), bitmap.getHeight(), 0.1f, resizedWidth, resizedHeight, left, top);
        }
    }

    private List<RotatedRect> decodeDetectionOutput(float[][][][] logits, int originalWidth, int originalHeight, float boxThreshold, int resizedWidth, int resizedHeight, int padLeft, int padTop) {
        // This function now implements post-processing for a DBNet-like model.
        // The model output is a probability map (logits).
        // Shape is (1, 1, H, W)
        float[][] logitsMap = logits[0][0];
        int height = logitsMap.length;
        int width = logitsMap[0].length;

        // --- 1. Create Mat from logits ---
        Mat logitsMat = new Mat(height, width, CvType.CV_32F);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                logitsMat.put(y, x, logitsMap[y][x]);
            }
        }
        
        // --- 2. Apply Sigmoid to get Probability Map ---
        Mat probMapMat = new Mat(height, width, CvType.CV_32F);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float logit = (float) logitsMat.get(y, x)[0];
                probMapMat.put(y, x, (float) (1.0 / (1.0 + Math.exp(-logit))));
            }
        }
        logitsMat.release();

        // --- 3. Binarization ---
        float binThresh = 0.3f; // Changed to match DBNet default
        Mat binaryMap = new Mat();
        Imgproc.threshold(probMapMat, binaryMap, binThresh, 255, Imgproc.THRESH_BINARY); // Binarize
        binaryMap.convertTo(binaryMap, CvType.CV_8U);

        // --- Apply morphological opening (erosion then dilation) like doctr DBNet ---
        Mat openingKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
        Imgproc.morphologyEx(binaryMap, binaryMap, Imgproc.MORPH_OPEN, openingKernel);
        openingKernel.release();

        // --- 4. Find Contours ---
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(binaryMap, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
        hierarchy.release();
        binaryMap.release();

        List<RotatedRect> validBoxes = new ArrayList<>();
        // Scaling factor to map coordinates from the resized image (inside the canvas) back to the original image
        // This scaling assumes the image was padded to fit the target size.
        double scaleW = (double) originalWidth / resizedWidth;
        double scaleH = (double) originalHeight / resizedHeight;

        // --- 5. Filter Contours & Unclip ---
        for (MatOfPoint contour : contours) {
            MatOfPoint2f contour2f = new MatOfPoint2f(contour.toArray());

            // --- 5a. Filter by area (from python code, min_size_box is small) ---
            if (Imgproc.contourArea(contour) < 2) {
                contour.release();
                contour2f.release();
                continue;
            }

            // --- 5b. Calculate confidence score from the probability map ---
            Mat mask = Mat.zeros(height, width, CvType.CV_8U);
            Imgproc.drawContours(mask, Collections.singletonList(contour), -1, new Scalar(255), -1); // Draw contour filled
            Scalar meanScoreScalar = Core.mean(probMapMat, mask);
            mask.release();
            float score = (float) meanScoreScalar.val[0];

            // --- 5c. Filter by confidence score (box_thresh) ---
            if (score < boxThreshold) {
                contour.release();
                contour2f.release();
                continue;
            }

            // --- 6. Unclip Polygon (approximating pyclipper with dilation) ---
            double area = Imgproc.contourArea(contour);
            double length = Imgproc.arcLength(contour2f, true);
            if (length == 0) { // Avoid division by zero
                contour.release();
                contour2f.release();
                continue;
            }

            // Unclip ratio for DBNet is 1.5
            double unclipRatio = 1.5; // Changed to match DBNet default
            double distance = (area * unclipRatio) / (length + 1e-6); // Added epsilon to avoid division by zero

            // Dilate the contour mask by 'distance' to unclip it.
            // Using a circular kernel for more even expansion, or a square for simpler approx.
            // A simple dilation kernel based on distance is an approximation of pyclipper.
            // The size needs to be odd, so 2*r+1
            int kernelSize = (int) Math.round(distance * 2) + 1; // Double distance for kernel diameter
            if (kernelSize < 3) kernelSize = 3; // Minimum kernel size
            if (kernelSize % 2 == 0) kernelSize++; // Ensure odd size

            Mat unclipKernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(kernelSize, kernelSize)); // Ellipse for smoother expansion

            Mat unclippedMask = Mat.zeros(height, width, CvType.CV_8U);
            Imgproc.drawContours(unclippedMask, Collections.singletonList(contour), -1, new Scalar(255), -1); // Draw original contour filled
            Imgproc.dilate(unclippedMask, unclippedMask, unclipKernel);
            unclipKernel.release();

            List<MatOfPoint> unclippedContours = new ArrayList<>();
            Imgproc.findContours(unclippedMask, unclippedContours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
            unclippedMask.release();

            RotatedRect box;
            if (!unclippedContours.isEmpty() && unclippedContours.get(0).total() > 0) {
                // Take the largest unclipped contour if multiple are found (pyclipper often returns one)
                MatOfPoint largestUnclippedContour = unclippedContours.get(0);
                for(int i = 1; i < unclippedContours.size(); i++){
                    if(Imgproc.contourArea(unclippedContours.get(i)) > Imgproc.contourArea(largestUnclippedContour)){
                        largestUnclippedContour = unclippedContours.get(i);
                    }
                }
                box = Imgproc.minAreaRect(new MatOfPoint2f(largestUnclippedContour.toArray()));
                for (MatOfPoint c : unclippedContours) c.release();
            } else {
                box = Imgproc.minAreaRect(contour2f); // Fallback to original contour's minAreaRect
            }

            // --- 7. Scale box to original image coordinates ---
            // Adjust center coordinates based on padding, then scale.
            double centerX = (box.center.x - padLeft) * scaleW; // Use scaleW for x
            double centerY = (box.center.y - padTop) * scaleH; // Use scaleH for y
            double w = box.size.width * scaleW; // Use scaleW for width
            double h = box.size.height * scaleH; // Use scaleH for height

            Point center = new Point(centerX, centerY);
            Size size = new Size(w, h);

            RotatedRect scaledBox = new RotatedRect(center, size, box.angle);
            // --- Bỏ các box nhỏ (width < 2 hoặc height < 2) ---
            Point[] pts = new Point[4];
            scaledBox.points(pts);
            // Calculate actual width and height of the rotated box based on points
            double boxWidth = Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y);
            double boxHeight = Math.hypot(pts[1].x - pts[2].x, pts[1].y - pts[2].y);
            if (boxWidth < 2 || boxHeight < 2) {
                contour.release();
                contour2f.release();
                continue;
            }
            validBoxes.add(scaledBox);
            contour.release();
            contour2f.release();
        }
        probMapMat.release();

        return validBoxes;
    }

    // A helper class to hold a bounding box and its original index, making it easier to manage.
    private static class WordBox {
        final RotatedRect box;
        final int originalIndex;

        WordBox(RotatedRect box, int originalIndex) {
            this.box = box;
            this.originalIndex = originalIndex;
        }
    }

    /**
     * Resolves a list of word boxes into lines and sub-lines, mimicking the logic from doctr's DocumentBuilder.
     * @param boxes A list of RotatedRect objects representing the detected words.
     * @return A list of lines, where each line is a list of WordBox objects.
     */
    private List<List<WordBox>> resolveLines(List<RotatedRect> boxes) {
        if (boxes == null || boxes.isEmpty()) {
            return new ArrayList<>();
        }

        // Wrap RotatedRects in WordBox objects to keep track of their original order/index if needed.
        List<WordBox> wordBoxes = new ArrayList<>();
        for (int i = 0; i < boxes.size(); i++) {
            wordBoxes.add(new WordBox(boxes.get(i), i));
        }

        // 1. Sort boxes from top to bottom, then left to right. This is a crucial step for line formation.
        wordBoxes.sort(Comparator.comparingDouble((WordBox wb) -> wb.box.center.y)
                .thenComparingDouble(wb -> wb.box.center.x));

        // 2. Calculate the median height of all boxes to use as a tolerance for line grouping.
        List<Double> heights = new ArrayList<>();
        for (WordBox wb : wordBoxes) {
            heights.add(wb.box.size.height);
        }
        Collections.sort(heights);
        double yMed = heights.isEmpty() ? 0 : (heights.size() % 2 == 0 ?
                (heights.get(heights.size() / 2 - 1) + heights.get(heights.size() / 2)) / 2.0 :
                heights.get(heights.size() / 2));

        if (yMed == 0) { // Avoid division by zero if heights are all zero.
            return new ArrayList<>();
        }

        List<List<WordBox>> lines = new ArrayList<>();
        if (wordBoxes.isEmpty()) {
            return lines;
        }

        // 3. Group boxes into lines based on vertical proximity.
        List<WordBox> currentLine = new ArrayList<>();
        currentLine.add(wordBoxes.get(0));
        double yCenterSum = wordBoxes.get(0).box.center.y;

        for (int i = 1; i < wordBoxes.size(); i++) {
            WordBox currentWord = wordBoxes.get(i);
            // Check the vertical distance between the current word and the center of the current line.
            double yDist = Math.abs(currentWord.box.center.y - (yCenterSum / currentLine.size()));

            if (yDist > yMed / 2) { // If the distance is too large, it's a new line.
                lines.addAll(resolveSubLines(currentLine)); // Process the completed line for horizontal breaks.
                currentLine.clear();
                yCenterSum = 0;
            }

            currentLine.add(currentWord);
            yCenterSum += currentWord.box.center.y;
        }

        // Don't forget the last line.
        if (!currentLine.isEmpty()) {
            lines.addAll(resolveSubLines(currentLine));
        }

        return lines;
    }

    /**
     * Splits a single line of words into multiple sub-lines if there are large horizontal gaps between them.
     * This is useful for handling multiple columns or paragraphs.
     * @param line A list of WordBox objects representing a single line.
     * @return A list of sub-lines.
     */
    private List<List<WordBox>> resolveSubLines(List<WordBox> line) {
        List<List<WordBox>> subLines = new ArrayList<>();
        if (line == null || line.isEmpty()) {
            return subLines;
        }

        // Sort words in the line horizontally from left to right.
        line.sort(Comparator.comparingDouble(wb -> wb.box.boundingRect().x));

        // Heuristic for paragraph break: use the median height of words in the line.
        // This is an adaptation of the python version's relative paragraph_break.
        List<Double> heights = new ArrayList<>();
        for (WordBox wb : line) {
            heights.add(wb.box.size.height);
        }
        Collections.sort(heights);
        double paragraphBreak = heights.isEmpty() ? 0 : (heights.size() % 2 == 0 ?
                (heights.get(heights.size() / 2 - 1) + heights.get(heights.size() / 2)) / 2.0 :
                heights.get(heights.size() / 2));


        if (line.size() < 2) {
            subLines.add(new ArrayList<>(line));
            return subLines;
        }

        List<WordBox> currentSubLine = new ArrayList<>();
        currentSubLine.add(line.get(0));

        for (int i = 1; i < line.size(); i++) {
            WordBox prevWord = currentSubLine.get(currentSubLine.size() - 1);
            WordBox currentWord = line.get(i);

            Rect prevBoxRect = prevWord.box.boundingRect();
            Rect currentBoxRect = currentWord.box.boundingRect();

            // Calculate the horizontal distance between the end of the previous word and the start of the current one.
            double dist = currentBoxRect.x - (prevBoxRect.x + prevBoxRect.width);

            if (dist > paragraphBreak) { // If the gap is larger than our threshold, start a new sub-line.
                subLines.add(new ArrayList<>(currentSubLine));
                currentSubLine.clear();
            }
            currentSubLine.add(currentWord);
        }

        subLines.add(new ArrayList<>(currentSubLine)); // Add the last sub-line.

        return subLines;
    }

    /**
     * Calculates the minimum area rotated rectangle that encloses all word boxes in a given line.
     * @param line A list of WordBox objects representing a single line.
     * @return A RotatedRect that tightly wraps the entire line.
     */
    private RotatedRect getEnclosingLineBox(List<WordBox> line) {
        if (line == null || line.isEmpty()) {
            return new RotatedRect(); // Return an empty rect if the line is empty
        }

        // Collect all corner points from all word boxes in the line
        List<Point> allPoints = new ArrayList<>();
        for (WordBox wordBox : line) {
            Point[] vertices = new Point[4];
            wordBox.box.points(vertices);
            Collections.addAll(allPoints, vertices);
        }

        // Use OpenCV to find the minimum area rectangle enclosing all the collected points
        MatOfPoint2f pointsMat = new MatOfPoint2f();
        pointsMat.fromList(allPoints);
        RotatedRect enclosingBox = Imgproc.minAreaRect(pointsMat);

        pointsMat.release(); // Release the native memory

        return enclosingBox;
    }

    private FloatBuffer preprocessImageForRecognition(Bitmap bitmap) {
        // The mean and std values are for the CRNN model, taken from the doctr source code.
        return preprocessImage(bitmap, 128, 32, new float[]{0.694f, 0.695f, 0.693f}, new float[]{0.299f, 0.296f, 0.301f});
    }

    private FloatBuffer preprocessImageForRecognition(Bitmap bitmap, int targetHeight, int targetWidth) {
        // This function uses a more generic pre-processing that also pads.
        // The mean and std are for models normalized to [-1, 1] or [0, 1]
        // For CRNN, it's normalized with specific mean/std values.
        return preprocessImage(bitmap, targetWidth, targetHeight, new float[]{0.694f, 0.695f, 0.693f}, new float[]{0.299f, 0.296f, 0.301f});
    }

    private FloatBuffer preprocessImageForDetection(Bitmap bitmap, int targetSize, int newWidth, int newHeight, float[] mean, float[] std) {
        Mat mat = new Mat();
        Utils.bitmapToMat(bitmap, mat);

        // Convert to RGB if it's RGBA
        Imgproc.cvtColor(mat, mat, Imgproc.COLOR_RGBA2RGB);

        // Resize the image, maintaining aspect ratio
        Mat resizedMat = new Mat();
        Imgproc.resize(mat, resizedMat, new Size(newWidth, newHeight), 0, 0, Imgproc.INTER_AREA);

        // Create a black canvas of the target size
        Mat canvasMat = Mat.zeros(targetSize, targetSize, CvType.CV_8UC3);

        // Copy the resized image onto the center of the canvas
        int top = (targetSize - newHeight) / 2;
        int left = (targetSize - newWidth) / 2;

        // --- START: ROI CRASH DIAGNOSIS ---
        Log.e("OcrCrashDebug", "--- ROI Pre-Check Values ---");
        Log.e("OcrCrashDebug", "targetSize = " + targetSize);
        Log.e("OcrCrashDebug", "newWidth = " + newWidth + ", newHeight = " + newHeight);
        Log.e("OcrCrashDebug", "left = " + left + ", top = " + top);
        Log.e("OcrCrashDebug", "Check (left + newWidth <= targetSize): " + (left + newWidth) + " <= " + targetSize + " ? --> " + ((left + newWidth) <= targetSize));
        Log.e("OcrCrashDebug", "Check (top + newHeight <= targetSize): " + (top + newHeight) + " <= " + targetSize + " ? --> " + ((top + newHeight) <= targetSize));
        // --- END: ROI CRASH DIAGNOSIS ---

        org.opencv.core.Rect roi = new org.opencv.core.Rect(left, top, newWidth, newHeight);
        Mat subview = canvasMat.submat(roi);
        resizedMat.copyTo(subview);

        // Convert to float and normalize
        canvasMat.convertTo(canvasMat, CvType.CV_32F, 1.0 / 255.0);
        Core.subtract(canvasMat, new Scalar(mean[0], mean[1], mean[2]), canvasMat);
        Core.divide(canvasMat, new Scalar(std[0], std[1], std[2]), canvasMat);

        // NCHW format for ONNX runtime
        float[] floatArray = new float[targetSize * targetSize * 3];
        canvasMat.get(0, 0, floatArray);
        FloatBuffer floatBuffer = FloatBuffer.allocate(targetSize * targetSize * 3);

        int stride = targetSize * targetSize;
        for (int i = 0; i < stride; i++) {
            floatBuffer.put(i, floatArray[i * 3]);             // R
            floatBuffer.put(i + stride, floatArray[i * 3 + 1]); // G
            floatBuffer.put(i + 2 * stride, floatArray[i * 3 + 2]); // B
        }
        floatBuffer.rewind();

        // Cleanup
        mat.release();
        resizedMat.release();
        canvasMat.release();
        subview.release();

        return floatBuffer;
    }

    private FloatBuffer preprocessImage(Bitmap bitmap, int targetWidth, int targetHeight, float[] mean, float[] std) {
        Mat mat = new Mat();
        Utils.bitmapToMat(bitmap, mat);
        Imgproc.cvtColor(mat, mat, Imgproc.COLOR_RGBA2RGB);

        // Resize and pad
        int h = mat.rows();
        int w = mat.cols();

        if (h == 0 || w == 0) {
            // Cannot process a zero-sized image. Return an empty buffer.
            return FloatBuffer.allocate(targetWidth * targetHeight * 3);
        }

        float scaleW = (float) targetWidth / w;
        float scaleH = (float) targetHeight / h;
        float scale = Math.min(scaleW, scaleH);

        int targetW = (int) (w * scale);
        int targetH = (int) (h * scale);

        // Safeguard against dimensions becoming zero after calculation.
        if (targetW <= 0) targetW = 1;
        if (targetH <= 0) targetH = 1;

        Mat resized = new Mat();
        Imgproc.resize(mat, resized, new Size(targetW, targetH), 0, 0, Imgproc.INTER_AREA);

        Mat finalMat = new Mat(targetHeight, targetWidth, resized.type(), new Scalar(0, 0, 0)); // Black padding
        int top = (targetHeight - targetH) / 2;
        int left = (targetWidth - targetW) / 2;
        org.opencv.core.Rect roi = new org.opencv.core.Rect(left, top, targetW, targetH);
        Mat subview = finalMat.submat(roi);
        resized.copyTo(subview);

        // Convert to float and normalize
        finalMat.convertTo(finalMat, CvType.CV_32F, 1.0 / 255.0);
        // Use per-channel mean and std. OpenCV Scalar is BGR, but our mat is RGB.
        // Let's assume the input mean/std are in RGB order.
        if (mean.length == 3 && std.length == 3) {
            Core.subtract(finalMat, new Scalar(mean[0], mean[1], mean[2]), finalMat);
            Core.divide(finalMat, new Scalar(std[0], std[1], std[2]), finalMat);
        } else { // Fallback for single value mean/std
            Core.subtract(finalMat, new Scalar(mean[0], mean[0], mean[0]), finalMat);
            Core.divide(finalMat, new Scalar(std[0], std[0], std[0]), finalMat);
        }


        // NCHW
        float[] floatArray = new float[targetWidth * targetHeight * 3];
        finalMat.get(0, 0, floatArray);
        FloatBuffer floatBuffer = FloatBuffer.allocate(targetWidth * targetHeight * 3);


        int stride = targetWidth * targetHeight;
        for (int i = 0; i < stride; i++) {
            floatBuffer.put(i, floatArray[i * 3]); // R
            floatBuffer.put(i + stride, floatArray[i * 3 + 1]); // G
            floatBuffer.put(i + 2 * stride, floatArray[i * 3 + 2]); // B
        }
        floatBuffer.rewind();

        mat.release();
        resized.release();
        finalMat.release();
        subview.release();

        return floatBuffer;
    }

    private void updateOcrResult(int position, Page page, boolean isProcessing) {
        if (position >= 0 && position < ocrResultList.size()) {
            OcrResult result = ocrResultList.get(position);
            result.setPage(page);
            result.setProcessing(isProcessing);
            result.setError(null); // Clear previous errors
            ocrResultAdapter.notifyItemChanged(position);

            // Sau khi OCR xong (isProcessing==false và page!=null), gửi lên backend
            if (!isProcessing && page != null && userId > 0) {
                String recognizedText = page.getContent();
                Uri imageUri = result.getImageUri();
                uploadOcrResultToBackend(userId, imageUri, recognizedText);
            }
        }
    }

    // Hàm upload kết quả OCR lên backend
    private void uploadOcrResultToBackend(int userId, Uri imageUri, String recognizedText) {
        try {
            InputStream inputStream = getContentResolver().openInputStream(imageUri);
            byte[] imageBytes = getBytes(inputStream);
            inputStream.close();

            OkHttpClient client = new OkHttpClient();
            RequestBody requestBody = new MultipartBody.Builder()
                    .setType(MultipartBody.FORM)
                    .addFormDataPart("user_id", String.valueOf(userId))
                    .addFormDataPart("recognized_text", recognizedText)
                    .addFormDataPart("image", "image.jpg",
                            RequestBody.create(imageBytes, MediaType.parse("image/jpeg")))
                    .build();

            Request request = new Request.Builder()
                    .url(BASE_URL + "/add_history")
                    .post(requestBody)
                    .build();

            new Thread(() -> {
                try (Response response = client.newCall(request).execute()) {
                    if (!response.isSuccessful()) {
                        Log.e("UploadHistory", "Failed: " + response.message());
                    } else {
                        Log.d("UploadHistory", "Success: " + response.body().string());
                    }
                } catch (Exception e) {
                    Log.e("UploadHistory", "Exception: " + e.getMessage());
                }
            }).start();

        } catch (Exception e) {
            Log.e("UploadHistory", "Exception: " + e.getMessage());
        }
    }

    private void updateOcrResult(int position, String errorMessage, boolean isProcessing, Bitmap preview) {
        if (position >= 0 && position < ocrResultList.size()) {
            OcrResult result = ocrResultList.get(position);
            result.setError(errorMessage);
            result.setPage(null); // Clear previous page results
            result.setProcessing(isProcessing);
            ocrResultAdapter.notifyItemChanged(position);
        }
    }

    private void triggerNextImageProcessing(String lang) {
        currentOcrIndex++;
        progressBar.setProgress((int) (((float) currentOcrIndex / imageUris.size()) * 100));
        processNextImage(lang);
    }
    
    private void setLocale(String langCode) {
        SharedPreferences.Editor editor = getSharedPreferences("settings", MODE_PRIVATE).edit();
        editor.putString("lang", langCode);
        editor.apply();
        Locale locale = new Locale(langCode);
        Locale.setDefault(locale);
        Configuration config = getResources().getConfiguration();
        config.setLocale(locale);
        getResources().updateConfiguration(config, getResources().getDisplayMetrics());
    }

    private void updateTexts() {
        setTitle(getString(R.string.app_name));
        tvSelectLanguage.setText("Select Language"); // Hardcoded to fix build error
        radioVietnamese.setText(getString(R.string.vietnamese));
        radioEnglish.setText(getString(R.string.english));
        tvDeleteInstruction.setText(getString(R.string.long_press_to_delete));
        // Update more texts if needed
    }

    private void setupNavigationDrawer() {
        NavigationView navigationView = findViewById(R.id.nav_view);
        View headerView = navigationView.getHeaderView(0);
        // ... find views in header ...
        
        if (isLoggedIn) {
            // Show user info, hide guest view
            guestViewNav.setVisibility(View.GONE);
            fetchHistory();
        } else {
            // Show guest view
            guestViewNav.setVisibility(View.VISIBLE);
        }

        btnLoginNav.setOnClickListener(v -> {
            // Handle login
        });
        
        btnNewSession.setOnClickListener(v -> startNewOcrSession());

        navigationView.setNavigationItemSelectedListener(item -> {
            // Handle navigation view item clicks here.
            drawerLayout.closeDrawers();
            return true;
        });
    }

    private void fetchHistory() {
        if (userId == -1) return;
        
        Request request = new Request.Builder()
                .url(BASE_URL + "/history/" + userId)
                .build();
                
        client.newCall(request).enqueue(new Callback() {
            @Override
            public void onFailure(@NonNull Call call, @NonNull IOException e) {
                runOnUiThread(() -> Toast.makeText(MainActivity.this, "Failed to fetch history.", Toast.LENGTH_SHORT).show());
            }

            @Override
            public void onResponse(@NonNull Call call, @NonNull Response response) throws IOException {
                if (response.isSuccessful()) {
                    final String jsonResponse = response.body().string();
                    try {
                        JSONObject jsonObj = new JSONObject(jsonResponse);
                        JSONArray sessionsArray = jsonObj.getJSONArray("sessions");
                        historySessionList.clear();
                        for (int i = 0; i < sessionsArray.length(); i++) {
                            JSONObject sessionObj = sessionsArray.getJSONObject(i);
                            String sessionName = sessionObj.getString("session_name");
                            String createdAt = sessionObj.getString("created_at");

                            List<HistoryItemDetail> details = new ArrayList<>();
                            List<Integer> imageIds = new ArrayList<>();
                            JSONArray imagesArray = sessionObj.getJSONArray("images");

                            for(int j=0; j<imagesArray.length(); j++){
                                JSONObject imageObj = imagesArray.getJSONObject(j);
                                int imageId = imageObj.getInt("image_id");
                                String base64 = imageObj.getString("base64_string");
                                String recognizedText = imageObj.optString("recognized_text", "N/A");
                                
                                Uri imageUri = saveBase64ImageToTempFile(base64, String.valueOf(imageId));
                                if(imageUri != null) {
                                    details.add(new HistoryItemDetail(imageUri, recognizedText));
                                    imageIds.add(imageId);
                                }
                            }
                            historySessionList.add(new HistorySession(sessionName, imagesArray.length(), details, imageIds));
                        }
                        
                        runOnUiThread(() -> historyAdapterNav.notifyDataSetChanged());
                        
                    } catch (JSONException e) {
                         runOnUiThread(() -> Toast.makeText(MainActivity.this, "Error parsing history.", Toast.LENGTH_SHORT).show());
                    }
                }
            }
        });
    }

    private Uri saveBase64ImageToTempFile(String base64String, String uniqueId) {
        try {
            final byte[] imageBytes = Base64.decode(base64String, Base64.DEFAULT);
            File outputDir = getCacheDir();
            File tempFile = File.createTempFile("history_" + uniqueId, ".jpg", outputDir);
            try (FileOutputStream fos = new FileOutputStream(tempFile)) {
                fos.write(imageBytes);
            }
            return Uri.fromFile(tempFile);
        } catch (IOException e) {
            Log.e("History", "Error saving base64 image to temp file", e);
            return null;
        }
    }
    
    @Override
    public void onImageClick(Uri imageUri) {
        int clickedPosition = imageUris.indexOf(imageUri);
        if (clickedPosition != -1) {
            Intent intent = new Intent(this, ImageViewerActivity.class);
            intent.putParcelableArrayListExtra("image_uris", new ArrayList<>(imageUris));
            intent.putExtra("selected_position", clickedPosition);
            imageViewerLauncher.launch(intent);
        }
    }
    
    @Override
    public void onSessionClick(HistorySession session) {
        imageUris.clear();
        ocrResultList.clear();
        
        if (session.getDetails() != null) {
            for(HistoryItemDetail detail : session.getDetails()){
                Uri uri = detail.getImageUri();
                if(uri != null){
                    imageUris.add(uri);
                    ocrResultList.add(new OcrResult(uri, detail.getRecognizedText(), false));
                }
            }
        }
        
        imageAdapter.notifyDataSetChanged();
        ocrResultAdapter.notifyDataSetChanged();
        updateDeleteInstructionVisibility();
        drawerLayout.closeDrawers();
    }
    
    @Override
    public void onDeleteSessionClick(HistorySession session, final int position) {
        new AlertDialog.Builder(this)
            .setTitle("Delete Session")
            .setMessage("Are you sure you want to delete this session?")
            .setPositiveButton("Delete", (dialog, which) -> {
                List<Integer> imageIds = session.getImageIds();
                if (imageIds != null && !imageIds.isEmpty()) {
                    sendDeleteSessionRequest(imageIds, position);
                } else {
                    // Fallback or error handling if needed, for now just remove from list
                    historySessionList.remove(position);
                    historyAdapterNav.notifyItemRemoved(position);
                    historyAdapterNav.notifyItemRangeChanged(position, historySessionList.size());
                }
            })
            .setNegativeButton("Cancel", null)
            .show();
    }

    private void sendDeleteSessionRequest(List<Integer> imageIds, final int position) {
        JSONObject json = new JSONObject();
        try {
            json.put("image_ids", new JSONArray(imageIds));
        } catch (JSONException e) {
            // Should not happen
            return;
        }

        RequestBody body = RequestBody.create(json.toString(), MediaType.get("application/json; charset=utf-8"));
        Request request = new Request.Builder()
                .url(BASE_URL + "/delete_images")
                .post(body)
                .build();
        
        client.newCall(request).enqueue(new Callback() {
            @Override
            public void onFailure(@NonNull Call call, @NonNull IOException e) {
                 runOnUiThread(() -> Toast.makeText(MainActivity.this, "Failed to delete session.", Toast.LENGTH_SHORT).show());
            }

            @Override
            public void onResponse(@NonNull Call call, @NonNull Response response) throws IOException {
                if (response.isSuccessful()) {
                    final String responseBody = response.body().string();
                    runOnUiThread(() -> {
                        try {
                            JSONObject json = new JSONObject(responseBody);
                            if (json.has("message") && json.getString("message").equals("Session deleted successfully")) {
                                historySessionList.remove(position);
                                historyAdapterNav.notifyItemRemoved(position);
                                historyAdapterNav.notifyItemRangeChanged(position, historySessionList.size());
                                Toast.makeText(MainActivity.this, getString(R.string.session_deleted), Toast.LENGTH_SHORT).show();
                            } else {
                                Toast.makeText(MainActivity.this, "Failed: " + json.optString("error", "Unknown error"), Toast.LENGTH_SHORT).show();
                            }
                        } catch (JSONException e) {
                            Log.e("MainActivity", "JSON parsing error after delete", e);
                            Toast.makeText(MainActivity.this, "Error parsing delete response", Toast.LENGTH_SHORT).show();
                        }
                    });
                } else {
                    runOnUiThread(() -> Toast.makeText(MainActivity.this, "Delete failed: " + response.message(), Toast.LENGTH_SHORT).show());
                }
            }
        });
    }

    private void startNewOcrSession() {
        // Clear current OCR data
        imageUris.clear();
        ocrResultList.clear();
        imageAdapter.notifyDataSetChanged();
        ocrResultAdapter.notifyDataSetChanged();
        currentOcrIndex = -1;
        stopOcrRequested = false;

        // Reset UI elements
        btnRunOcr.setVisibility(View.VISIBLE);
        btnStopOcr.setVisibility(View.GONE);
        progressBar.setVisibility(View.GONE);

        // Close the drawer
        if (drawerLayout.isDrawerOpen(findViewById(R.id.nav_view))) {
            drawerLayout.closeDrawer(findViewById(R.id.nav_view));
        }
        Toast.makeText(this, getString(R.string.new_session_started), Toast.LENGTH_SHORT).show();
    }
    
    private void adjustNavDrawerForStatusBar() {
        // This is a common method to adjust navigation drawer padding for transparent status bars.
        // BUG FIX: Add padding to the top of the navigation view to avoid being obscured by the system status bar.
        int statusBarHeight = 0;
        int resourceId = getResources().getIdentifier("status_bar_height", "dimen", "android");
        if (resourceId > 0) {
            statusBarHeight = getResources().getDimensionPixelSize(resourceId);
        }
        NavigationView navigationView = findViewById(R.id.nav_view);
        navigationView.setPadding(navigationView.getPaddingLeft(), navigationView.getPaddingTop() + statusBarHeight, navigationView.getPaddingRight(), navigationView.getPaddingBottom());
    }
    
    private void updateDeleteInstructionVisibility() {
        if (!imageUris.isEmpty()) {
            tvDeleteInstruction.setVisibility(View.VISIBLE);
        } else {
            tvDeleteInstruction.setVisibility(View.GONE);
        }
    }

    public byte[] getBytes(InputStream inputStream) throws IOException {
        ByteArrayOutputStream byteBuffer = new ByteArrayOutputStream();
        int bufferSize = 1024;
        byte[] buffer = new byte[bufferSize];

        int len;
        while ((len = inputStream.read(buffer)) != -1) {
            byteBuffer.write(buffer, 0, len);
        }
        return byteBuffer.toByteArray();
    }

    private void showStopOcrConfirmationDialog() {
        if (!progressBar.isIndeterminate()) {
            return; // Not running
        }
        new AlertDialog.Builder(this)
                .setTitle("Stop OCR")
                .setMessage("Are you sure you want to stop the current OCR process?")
                .setPositiveButton("Stop", (dialog, which) -> {
                    stopOcrRequested = true;
                    if(currentOcrCall != null && !currentOcrCall.isCanceled()){
                        currentOcrCall.cancel();
                    }
                })
                .setNegativeButton("Cancel", null)
                .show();
    }
    
    private void updateOcrUiState(boolean isRunning) {
        progressBar.setIndeterminate(isRunning);
        progressBar.setVisibility(isRunning ? View.VISIBLE : View.GONE);
        if(isRunning){
            progressBar.setProgress(0);
        }
        btnRunOcr.setEnabled(!isRunning);
        btnStopOcr.setEnabled(isRunning);
        btnExport.setEnabled(!isRunning && !ocrResultList.isEmpty());
        fab.setEnabled(!isRunning);
    }
    
    @Override
    public void onImageLongClick(Uri imageUri) {
        new AlertDialog.Builder(this)
            .setTitle("Delete Image")
            .setMessage("Are you sure you want to remove this image from the list?")
            .setPositiveButton("Delete", (dialog, which) -> {
                int position = imageUris.indexOf(imageUri);
                if (position != -1) {
                    imageUris.remove(position);
                    imageAdapter.notifyItemRemoved(position);
                    imageAdapter.notifyItemRangeChanged(position, imageUris.size());
                    updateDeleteInstructionVisibility();
                    
                    // Also remove the corresponding ocr result if it exists
                    if(position < ocrResultList.size()){
                        ocrResultList.remove(position);
                        ocrResultAdapter.notifyItemRemoved(position);
                        ocrResultAdapter.notifyItemRangeChanged(position, ocrResultList.size());
                    }
                }
            })
            .setNegativeButton("Cancel", null)
            .show();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        // Close OrtSession and OrtEnvironment to free resources
        try {
            if (vietOcr != null) vietOcr.close();
            if (ortSession != null) ortSession.close();
            if (detectionSession != null) detectionSession.close();
            if (englishRecognitionSession != null) englishRecognitionSession.close();
            if (ortEnv != null) ortEnv.close();
        } catch (Exception e) {
            Log.e("MainActivity", "Error closing ONNX Runtime resources.", e);
        }
    }
    
    private String assetFilePath(String assetName) throws IOException {
        File file = new File(this.getCacheDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = this.getAssets().open(assetName);
             OutputStream os = new FileOutputStream(file)) {
            byte[] buffer = new byte[4 * 1024];
            int read;
            while ((read = is.read(buffer)) != -1) {
                os.write(buffer, 0, read);
            }
            os.flush();
        }
        return file.getAbsolutePath();
    }

    private void exportRecognizedText() {
        if (ocrResultList == null || ocrResultList.stream().allMatch(r -> r.getPage() == null || r.getPage().getContent() == null || r.getPage().getContent().isEmpty())) {
            Toast.makeText(this, "No OCR results to export.", Toast.LENGTH_SHORT).show();
            return;
        }

        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE}, REQUEST_WRITE_STORAGE_PERMISSION);
                return; 
            }
        }
        
        showExportFormatDialog();
    }
    
    private void showExportFormatDialog() {
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setTitle("Choose Export Format");
        String[] formats = {"PDF", "DOCX"};
        builder.setItems(formats, (dialog, which) -> {
            if (which == 0) {
                exportToFile("pdf");
            } else {
                exportToFile("docx");
            }
        });
        builder.show();
    }
    
    private void exportToFile(String format) {
        StringBuilder fullText = new StringBuilder();
        for (OcrResult result : ocrResultList) {
            if (result.getPage() != null && result.getPage().getContent() != null) {
                fullText.append(result.getPage().getContent()).append("\n\n");
            }
        }
    
        if (fullText.length() == 0) {
            Toast.makeText(this, "No text to export.", Toast.LENGTH_SHORT).show();
            return;
        }
    
        if ("pdf".equals(format)) {
            Intent intent = new Intent(Intent.ACTION_CREATE_DOCUMENT);
            intent.addCategory(Intent.CATEGORY_OPENABLE);
            intent.setType("application/pdf");
            intent.putExtra(Intent.EXTRA_TITLE, "OCR_Result.pdf");
            startActivityForResult(intent, REQUEST_CODE_CREATE_PDF);
            pendingExportPdfUri = null; // reset
            return;
        } else if ("docx".equals(format)) {
            Intent intent = new Intent(Intent.ACTION_CREATE_DOCUMENT);
            intent.addCategory(Intent.CATEGORY_OPENABLE);
            intent.setType("application/vnd.openxmlformats-officedocument.wordprocessingml.document");
            intent.putExtra(Intent.EXTRA_TITLE, "OCR_Result.docx");
            startActivityForResult(intent, REQUEST_CODE_CREATE_DOCX);
            return;
        }
    
        try {
            String fileName = "OCR_Result_" + System.currentTimeMillis() + "." + format;
            ContentValues values = new ContentValues();
            values.put(MediaStore.MediaColumns.DISPLAY_NAME, fileName);
            values.put(MediaStore.MediaColumns.IS_PENDING, 1);
    
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                values.put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS);
            }
    
            Uri collection = MediaStore.Files.getContentUri("external");
            Uri itemUri = getContentResolver().insert(collection, values);
    
            if (itemUri == null) throw new IOException("Failed to create new MediaStore entry.");
    
            try (OutputStream os = getContentResolver().openOutputStream(itemUri)) {
                if (os == null) throw new IOException("Failed to get output stream.");
    
                if ("docx".equals(format)) {
                    values.put(MediaStore.MediaColumns.MIME_TYPE, "application/vnd.openxmlformats-officedocument.wordprocessingml.document");
                    XWPFDocument document = new XWPFDocument();
                    String[] lines = fullText.toString().split("\n");
                    for (String line : lines) {
                        XWPFParagraph paragraph = document.createParagraph();
                        XWPFRun run = paragraph.createRun();
                        run.setText(line);
                    }
                    document.write(os);
                }
            }
            
            values.clear();
            values.put(MediaStore.MediaColumns.IS_PENDING, 0);
            getContentResolver().update(itemUri, values, null, null);
    
            Toast.makeText(this, "Exported successfully to Downloads", Toast.LENGTH_LONG).show();
    
        } catch (Exception e) {
            Log.e("MainActivity", "Error exporting file", e);
            Toast.makeText(this, "Error exporting file: " + e.getMessage(), Toast.LENGTH_LONG).show();
        }
    }

    private void saveSessionToHistory() {
        if (userId == -1 || ocrResultList.isEmpty()) {
            return;
        }

        JSONObject sessionData = new JSONObject();
        try {
            sessionData.put("user_id", userId);
            JSONArray imagesArray = new JSONArray();

            for (OcrResult ocrResult : ocrResultList) {
                if(ocrResult.isProcessing()) continue; // Don't save incomplete results

                try {
                    // Convert Uri to Base64
                    InputStream inputStream = getContentResolver().openInputStream(ocrResult.getImageUri());
                    byte[] imageBytes = getBytes(inputStream);
                    String base64Image = Base64.encodeToString(imageBytes, Base64.DEFAULT);

                    JSONObject imageData = new JSONObject();
                    imageData.put("base64_string", base64Image);
                    imageData.put("recognized_text", ocrResult.getText());
                    imagesArray.put(imageData);
                } catch (IOException e) {
                    Log.e("MainActivity", "Failed to process image for history", e);
                }
            }
            sessionData.put("images", imagesArray);
            sessionData.put("session_name", "OCR Session " + System.currentTimeMillis());

        } catch (JSONException e) {
            Log.e("MainActivity", "Failed to create JSON for history", e);
            return;
        }

        RequestBody body = RequestBody.create(sessionData.toString(), MediaType.get("application/json; charset=utf-8"));
        Request request = new Request.Builder()
                .url(BASE_URL + "/add_session") // Assuming this endpoint
                .post(body)
                .build();

        client.newCall(request).enqueue(new Callback() {
            @Override
            public void onFailure(@NonNull Call call, @NonNull IOException e) {
                runOnUiThread(() -> Toast.makeText(MainActivity.this, "Failed to save session to history.", Toast.LENGTH_SHORT).show());
            }

            @Override
            public void onResponse(@NonNull Call call, @NonNull Response response) throws IOException {
                if (response.isSuccessful()) {
                    runOnUiThread(() -> {
                        Toast.makeText(MainActivity.this, "Session saved.", Toast.LENGTH_SHORT).show();
                        // Optionally, refresh history view
                        fetchHistory();
                    });
                } else {
                    final String errorBody = response.body().string();
                    runOnUiThread(() -> Toast.makeText(MainActivity.this, "Failed to save session: " + errorBody, Toast.LENGTH_SHORT).show());
                }
            }
        });
    }

    /**
     * Groups a list of lines into blocks of text (paragraphs) using hierarchical clustering.
     * This method mimics the _resolve_blocks logic from doctr's DocumentBuilder.
     * @param lines The list of Line objects to group.
     * @return A list of Block objects.
     */
    private List<Block> resolveBlocks(List<Line> lines) {
        List<Block> blocks = new ArrayList<>();
        for (Line line : lines) {
            // Mỗi block chỉ chứa 1 line, geometry là geometry của line
            blocks.add(new Block(Collections.singletonList(line), line.getGeometry()));
        }
        return blocks;
    }
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == REQUEST_CODE_CREATE_PDF && resultCode == RESULT_OK && data != null) {
            Uri uri = data.getData();
            if (uri != null) {
                try (OutputStream os = getContentResolver().openOutputStream(uri)) {
                    Document document = new Document();
                    PdfWriter.getInstance(document, os);
                    document.open();
                    try {
                        InputStream fontStream = getAssets().open("fonts/font-times-new-roman/times_400.ttf");
                        byte[] fontBytes = getBytes(fontStream);
                        BaseFont bf = BaseFont.createFont("times_400.ttf", BaseFont.IDENTITY_H, BaseFont.EMBEDDED, false, fontBytes, null);
                        Font font = new Font(bf, 12);
                        document.add(new Paragraph(fullTextForExport(), font));
                    } catch (IOException e) {
                        document.add(new Paragraph(fullTextForExport()));
                    }
                    document.close();
                    Toast.makeText(this, "Exported successfully", Toast.LENGTH_LONG).show();
                } catch (Exception e) {
                    Toast.makeText(this, "Error exporting file: " + e.getMessage(), Toast.LENGTH_LONG).show();
                }
            }
        } else if (requestCode == REQUEST_CODE_CREATE_DOCX && resultCode == RESULT_OK && data != null) {
            Uri uri = data.getData();
            if (uri != null) {
                try (OutputStream os = getContentResolver().openOutputStream(uri)) {
                    XWPFDocument document = new XWPFDocument();
                    String[] lines = fullTextForExport().split("\n");
                    for (String line : lines) {
                        XWPFParagraph paragraph = document.createParagraph();
                        XWPFRun run = paragraph.createRun();
                        run.setText(line);
                    }
                    document.write(os);
                    Toast.makeText(this, "Exported successfully", Toast.LENGTH_LONG).show();
                } catch (Exception e) {
                    Toast.makeText(this, "Error exporting file: " + e.getMessage(), Toast.LENGTH_LONG).show();
                }
            }
        }
    }

    private String fullTextForExport() {
        StringBuilder fullText = new StringBuilder();
        for (OcrResult result : ocrResultList) {
            if (result.getPage() != null && result.getPage().getContent() != null) {
                fullText.append(result.getPage().getContent()).append("\n\n");
            }
        }
        return fullText.toString();
    }
}