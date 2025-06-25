package com.example.myocr;

import android.Manifest;
import android.content.ContentValues;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.Configuration;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
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

public class MainActivity extends AppCompatActivity implements ImageAdapter.OnImageClickListener, HistoryAdapter.OnHistorySessionInteractionListener {
    private static final int REQUEST_CAMERA_PERMISSION = 100;
    private static final int REQUEST_WRITE_STORAGE_PERMISSION = 102; // For export
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
    private static final String BASE_URL = "https://7c2c-2405-4803-f801-12a0-1883-6ffe-89a4-5660.ngrok-free.app"; // IMPORTANT: Use your actual server URL

    static {
        if(OpenCVLoader.initDebug()){
            Log.d("MainActivity", "OpenCV is loaded");
        } else {
            Log.e("MainActivity", "OpenCV is not loaded");
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        // Initialize ONNX Runtime
        try {
            ortEnv = OrtEnvironment.getEnvironment();

            byte[] vietocrModelData = readBytesFromAsset("vietocr_vgg_seq2seq_opset14.onnx");
            ortSession = ortEnv.createSession(vietocrModelData);

            byte[] detectionModelData = readBytesFromAsset("fast_small_detection.onnx");
            detectionSession = ortEnv.createSession(detectionModelData);

            byte[] englishRecognitionModelData = readBytesFromAsset("vitstr_small_recognition.onnx");
            englishRecognitionSession = ortEnv.createSession(englishRecognitionModelData);

            Log.d("MainActivity", "All ONNX models loaded successfully.");

            // Initialize Vocabularies
            // Vietnamese vocabulary from base.yml
            String vietnameseChars = "aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợƠpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỨvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!\\\"#$%&\'()*+,-./:;<=>?@[\\\\]^_`{|}~ ";
            vietnameseVocab = new Vocab(vietnameseChars);

            // English vocabulary (common characters, you might need to adjust this based on vitstr_small_recognition's actual vocab)
            String englishChars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\\\"#$%&\'()*+,-./:;<=>?@[\\\\]^_`{|}~ ";
            englishVocab = new Vocab(englishChars);

        } catch (Exception e) {
            Log.e("MainActivity", "Error loading ONNX models or vocab: " + e.getMessage());
            Toast.makeText(this, "Error loading ONNX models or vocab: " + e.getMessage(), Toast.LENGTH_LONG).show();
        }

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

        // Setup for the selected images RecyclerView
        imageAdapter = new ImageAdapter(this, imageUris, this);
        imageRecyclerView.setLayoutManager(new LinearLayoutManager(this, LinearLayoutManager.HORIZONTAL, false));
        imageRecyclerView.setAdapter(imageAdapter);

        // Setup for the OCR results RecyclerView
        ocrResultRecyclerView = findViewById(R.id.ocrResultRecyclerView);
        ocrResultAdapter = new OcrResultAdapter(this, ocrResultList);
        ocrResultRecyclerView.setLayoutManager(new LinearLayoutManager(this));
        ocrResultRecyclerView.setAdapter(ocrResultAdapter);

        pickImageLauncher = registerForActivityResult(
                new ActivityResultContracts.StartActivityForResult(),
                result -> {
                    if (result.getResultCode() == RESULT_OK && result.getData() != null) {
                        Intent data = result.getData();
                        if (data.getClipData() != null) {
                            int count = data.getClipData().getItemCount();
                            for (int i = 0; i < count; i++) {
                                Uri imageUri = data.getClipData().getItemAt(i).getUri();
                                if (imageUri != null) {
                                    imageUris.add(imageUri);
                                }
                            }
                            imageAdapter.setImageUris(imageUris); // Update for multi-select
                        } else if (data.getData() != null) {
                            Uri imageUri = data.getData();
                            // Immediately launch ImageViewerActivity for a new image
                            Intent intent = new Intent(MainActivity.this, ImageViewerActivity.class);
                            intent.setData(imageUri);
                            intent.putExtra("image_position", -1); // -1 indicates a new image
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
                            imageUris.add(cameraImageUri);
                            imageAdapter.setImageUris(imageUris);
                        }
                    }
                }
        );

        imageViewerLauncher = registerForActivityResult(
                new ActivityResultContracts.StartActivityForResult(),
                result -> {
                    if (result.getResultCode() == RESULT_OK && result.getData() != null) {
                        Uri editedImageUri = result.getData().getData();
                        int position = result.getData().getIntExtra("image_position", -1);

                        if (editedImageUri != null) {
                            if (position > -1 && position < imageUris.size()) {
                                // Replace existing image
                                imageUris.set(position, editedImageUri);
                            } else {
                                // Add new image
                                imageUris.add(editedImageUri);
                            }
                            imageAdapter.setImageUris(imageUris);
                        }
                    }
                }
        );

        fab.setOnClickListener(v -> showImageSourceDialog());

        btnRunOcr.setOnClickListener(v -> runOcrOnImages());
        btnStopOcr.setOnClickListener(v -> showStopOcrConfirmationDialog());
        btnExport.setOnClickListener(v -> exportRecognizedText());

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
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, 101);
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
        Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        intent.setType("image/*");
        intent.putExtra(Intent.EXTRA_ALLOW_MULTIPLE, true);
        pickImageLauncher.launch(intent);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_CAMERA_PERMISSION) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                openCamera();
            }
        } else if (requestCode == 101) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                // Permission granted, retry the operation
            } else {
                Toast.makeText(this, "Permission denied to read external storage", Toast.LENGTH_SHORT).show();
            }
        } else if (requestCode == REQUEST_WRITE_STORAGE_PERMISSION) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                Toast.makeText(this, "Storage permission granted. Please try exporting again.", Toast.LENGTH_SHORT).show();
            } else {
                Toast.makeText(this, "Storage permission denied. Cannot export file.", Toast.LENGTH_SHORT).show();
            }
        }
    }

    private void runOcrOnImages() {
        if (imageUris.isEmpty()) {
            Toast.makeText(this, R.string.please_select_image, Toast.LENGTH_SHORT).show();
            return;
        }

        stopOcrRequested = false; // Reset stop flag
        currentOcrIndex = -1;
        ocrResultList.clear();
        for (Uri uri : imageUris) {
            ocrResultList.add(new OcrResult(uri, "Queued...", false));
        }
        ocrResultAdapter.setOcrResults(ocrResultList);

        updateOcrUiState(true); // Show progress and stop button
        processNextImage();
    }

    private void processNextImage() {
        // Check if the process was requested to stop
        if (stopOcrRequested) {
            updateOcrUiState(false); // Ensure UI is reset
            return;
        }

        currentOcrIndex++;
        if (currentOcrIndex >= imageUris.size()) {
            Toast.makeText(this, "All images processed.", Toast.LENGTH_SHORT).show();
            updateOcrUiState(false); // Hide progress bar and stop button
            if (isLoggedIn) fetchHistory(); // Refresh history
            return;
        }

        // Update progress bar
        progressBar.setMax(imageUris.size());
        progressBar.setProgress(currentOcrIndex + 1);

        Uri imageUri = imageUris.get(currentOcrIndex);
        updateOcrResult(currentOcrIndex, getString(R.string.processing), true);
        performOcrForImage(imageUri, currentOcrIndex, userId);
    }

    private void exportRecognizedText() {
        if (ocrResultList.isEmpty() || getCombinedOcrText().isEmpty()) {
            Toast.makeText(this, "Không có văn bản để xuất.", Toast.LENGTH_SHORT).show();
            return;
        }

        final String[] formats = {"PDF", "DOCX"};
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setTitle("Chọn định dạng xuất");
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
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE}, REQUEST_WRITE_STORAGE_PERMISSION);
                return;
            }
        }

        if ("pdf".equals(format)) {
            createPdf();
        } else if ("docx".equals(format)) {
            createDocx();
        }
    }

    private String getCombinedOcrText() {
        StringBuilder fullText = new StringBuilder();
        for (OcrResult result : ocrResultList) {
            if (result.getRecognizedText() != null && !result.getRecognizedText().isEmpty() &&
                    !result.getRecognizedText().equals("Queued...") && !result.getRecognizedText().equals("Đang xử lý...") &&
                    !result.getRecognizedText().contains("Error") && !result.getRecognizedText().contains("Lỗi")) {
                fullText.append(result.getRecognizedText()).append("\n\n---\n\n");
            }
        }
        return fullText.toString();
    }

    private void createPdf() {
        String ocrText = getCombinedOcrText();
        String fileName = "OCR_Result_" + System.currentTimeMillis() + ".pdf";

        ContentValues contentValues = new ContentValues();
        contentValues.put(MediaStore.MediaColumns.DISPLAY_NAME, fileName);
        contentValues.put(MediaStore.MediaColumns.MIME_TYPE, "application/pdf");
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            contentValues.put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS);
        }

        Uri uri = getContentResolver().insert(Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q
                ? MediaStore.Downloads.EXTERNAL_CONTENT_URI
                : MediaStore.Files.getContentUri("external"), contentValues);

        if (uri != null) {
            try (OutputStream outputStream = getContentResolver().openOutputStream(uri)) {
                Document document = new Document();
                PdfWriter.getInstance(document, outputStream);
                document.open();

                try {
                    BaseFont baseFont = BaseFont.createFont("assets/fonts/Roboto_Condensed-BoldItalic.ttf", BaseFont.IDENTITY_H, BaseFont.EMBEDDED);
                    Font vietnameseFont = new Font(baseFont, 12);
                    document.add(new Paragraph(ocrText, vietnameseFont));
                } catch (Exception e) {
                    Log.e("PDF_FONT", "Lỗi tải font, sử dụng font mặc định.", e);
                    document.add(new Paragraph(ocrText));
                }
                
                document.close();
                Toast.makeText(this, "Đã lưu PDF vào thư mục Downloads.", Toast.LENGTH_LONG).show();
            } catch (Exception e) {
                Log.e("ExportPDF", "Không thể tạo tệp PDF", e);
                Toast.makeText(this, "Lỗi khi lưu tệp PDF.", Toast.LENGTH_SHORT).show();
            }
        }
    }

    private void createDocx() {
        String ocrText = getCombinedOcrText();
        String fileName = "OCR_Result_" + System.currentTimeMillis() + ".docx";

        ContentValues contentValues = new ContentValues();
        contentValues.put(MediaStore.MediaColumns.DISPLAY_NAME, fileName);
        contentValues.put(MediaStore.MediaColumns.MIME_TYPE, "application/vnd.openxmlformats-officedocument.wordprocessingml.document");
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            contentValues.put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS);
        }

        Uri uri = getContentResolver().insert(Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q
                ? MediaStore.Downloads.EXTERNAL_CONTENT_URI
                : MediaStore.Files.getContentUri("external"), contentValues);

        if (uri != null) {
            try (OutputStream outputStream = getContentResolver().openOutputStream(uri);
                 XWPFDocument document = new XWPFDocument()) {

                String[] lines = ocrText.split("\n");
                for (String line : lines) {
                    XWPFParagraph paragraph = document.createParagraph();
                    XWPFRun run = paragraph.createRun();
                    run.setText(line);
                }

                document.write(outputStream);
                Toast.makeText(this, "Đã lưu DOCX vào thư mục Downloads.", Toast.LENGTH_LONG).show();
            } catch (Exception e) {
                Log.e("ExportDOCX", "Không thể tạo tệp DOCX", e);
                Toast.makeText(this, "Lỗi khi lưu tệp DOCX.", Toast.LENGTH_SHORT).show();
            }
        }
    }

    private FloatBuffer preprocessImageForRecognition(Bitmap bitmap) {
        // Resize bitmap to 32x128 (Height x Width) as required by the model
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, 128, 32, true);

        int width = resizedBitmap.getWidth();
        int height = resizedBitmap.getHeight();
        FloatBuffer floatBuffer = FloatBuffer.allocate(width * height * 3); // RGB channels
        int[] intValues = new int[width * height];
        resizedBitmap.getPixels(intValues, 0, width, 0, 0, width, height);

        for (int i = 0; i < intValues.length; ++i) {
            final int val = intValues[i];
            // Normalize to -1 to 1 range and handle channels
            floatBuffer.put((((float) ((val >> 16) & 0xFF)) / 255.0f - 0.5f) / 0.5f); // Red
            floatBuffer.put((((float) ((val >> 8) & 0xFF)) / 255.0f - 0.5f) / 0.5f);  // Green
            floatBuffer.put((((float) (val & 0xFF)) / 255.0f - 0.5f) / 0.5f);         // Blue
        }
        floatBuffer.rewind();
        return floatBuffer;
    }

    private FloatBuffer preprocessImageForDetection(Bitmap bitmap) {
        // Resize bitmap to a common detection model input size, e.g., 640x640
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, 640, 640, true);

        int width = resizedBitmap.getWidth();
        int height = resizedBitmap.getHeight();
        FloatBuffer floatBuffer = FloatBuffer.allocate(width * height * 3); // RGB channels
        int[] intValues = new int[width * height];
        resizedBitmap.getPixels(intValues, 0, width, 0, 0, width, height);

        // Normalize pixel values to 0-1 range
        for (int i = 0; i < intValues.length; ++i) {
            final int val = intValues[i];
            floatBuffer.put(((float) ((val >> 16) & 0xFF)) / 255.0f); // Red
            floatBuffer.put(((float) ((val >> 8) & 0xFF)) / 255.0f);  // Green
            floatBuffer.put(((float) (val & 0xFF)) / 255.0f);         // Blue
        }
        floatBuffer.rewind();
        return floatBuffer;
    }

    private void performOcrForImage(Uri imageUri, final int position, int userId) {
        // Add null checks for ONNX Runtime sessions
        if (ortSession == null || detectionSession == null || englishRecognitionSession == null) {
            runOnUiThread(() -> updateOcrResult(position, getString(R.string.failed_to_process_image, "ONNX Runtime sessions not initialized. Please restart the app."), false));
            triggerNextImageProcessing();
            return;
        }

        try (InputStream iStream = getContentResolver().openInputStream(imageUri)) {
            if (iStream == null) throw new IOException("Unable to open InputStream.");
            Bitmap originalBitmap = BitmapFactory.decodeStream(iStream);

            if (originalBitmap == null) {
                runOnUiThread(() -> updateOcrResult(position, getString(R.string.failed_to_process_image, "Could not decode image."), false));
                triggerNextImageProcessing();
                return;
            }

            // --- TEXT DETECTION STEP ---
            FloatBuffer detectionInputBuffer = preprocessImageForDetection(originalBitmap);
            long[] detectionInputShape = {1, 3, 640, 640}; // Batch size 1, 3 channels, 640 height, 640 width
            java.util.Map<String, OnnxTensor> detectionInputs = new java.util.HashMap<>();
            detectionInputs.put(new ArrayList<>(detectionSession.getInputNames()).get(0), OnnxTensor.createTensor(ortEnv, detectionInputBuffer, detectionInputShape));

            // Run detection model
            OrtSession.Result detectionResult = detectionSession.run(detectionInputs);

            // Process detection output
            OnnxTensor detectionOutput = (OnnxTensor) detectionResult.get(new ArrayList<>(detectionSession.getOutputNames()).get(0)).get();
            FloatBuffer detectionOutputBuffer = detectionOutput.getFloatBuffer();
            // TODO: Parse detectionResult to get bounding boxes of text
            // For now, we will assume one full image detection for simplicity.
            // This part will be expanded in the next steps.
            detectionResult.close(); // Close detection result

            // --- RECOGNITION STEP (using the whole image as a single text line for now) ---
            // This part will be iterated over detected bounding boxes in the next steps.
            FloatBuffer recognitionInputData = preprocessImageForRecognition(originalBitmap);

            long[] recognitionInputShape = {1, 3, 32, 128}; // Batch size 1, 3 channels, 32 height, 128 width
            java.util.Map<String, OnnxTensor> recognitionInputs = new java.util.HashMap<>();
            recognitionInputs.put(new ArrayList<>(ortSession.getInputNames()).get(0), OnnxTensor.createTensor(ortEnv, recognitionInputData, recognitionInputShape));

            // Determine which recognition model to use based on language selection
            OrtSession currentRecognitionSession;
            Vocab currentVocab;
            if (radioVietnamese.isChecked()) {
                currentRecognitionSession = ortSession;
                currentVocab = vietnameseVocab;
            } else {
                currentRecognitionSession = englishRecognitionSession;
                currentVocab = englishVocab;
            }

            // Run recognition model
            OrtSession.Result recognitionResult = currentRecognitionSession.run(recognitionInputs);
            OnnxTensor recognitionOutput = (OnnxTensor) recognitionResult.get(new ArrayList<>(currentRecognitionSession.getOutputNames()).get(0)).get();
            long[] outputShape = (long[]) recognitionOutput.getInfo().getShape(); // Shape is [batch_size, seq_len, vocab_size]
            float[] outputData = recognitionOutput.getFloatBuffer().array();

            // Decode the logits to text
            int seqLen = (int) outputShape[1];
            int vocabSize = (int) outputShape[2];
            int[] predictedCharIds = new int[seqLen];

            for (int i = 0; i < seqLen; i++) {
                int bestCharIndex = -1;
                float maxLogit = Float.MIN_VALUE;
                for (int j = 0; j < vocabSize; j++) {
                    float currentLogit = outputData[i * vocabSize + j];
                    if (currentLogit > maxLogit) {
                        maxLogit = currentLogit;
                        bestCharIndex = j;
                    }
                }
                predictedCharIds[i] = bestCharIndex;
            }

            String recognizedText = currentVocab.decode(predictedCharIds);

            runOnUiThread(() -> updateOcrResult(position, recognizedText, false));
            recognitionResult.close(); // Close the recognition result to free resources

        } catch (Exception e) {
            Log.e("MainActivity", "Error during ONNX inference: " + e.getMessage());
            runOnUiThread(() -> updateOcrResult(position, getString(R.string.failed_to_process_image, e.getMessage()), false));
        } finally {
            // Make sure to close current OCR call if it was an HTTP one (before modification)
            if (currentOcrCall != null && !currentOcrCall.isCanceled()) {
                currentOcrCall.cancel();
            }
            triggerNextImageProcessing();
        }
    }

    private void updateOcrResult(int position, String text, boolean isProcessing) {
        if (position >= 0 && position < ocrResultList.size()) {
            OcrResult result = ocrResultList.get(position);
            result.setRecognizedText(text);
            result.setProcessing(isProcessing);
            ocrResultAdapter.notifyItemChanged(position);
        }
    }

    private void triggerNextImageProcessing() {
        runOnUiThread(this::processNextImage);
    }

    private void setLocale(String langCode) {
        Locale locale = new Locale(langCode);
        Locale.setDefault(locale);
        Configuration config = new Configuration();
        config.setLocale(locale);
        getResources().updateConfiguration(config, getResources().getDisplayMetrics());
    }

    private void updateTexts() {
        btnRunOcr.setText(getString(R.string.run_ocr));
        btnExport.setText(getString(R.string.export));
        tvSelectLanguage.setText(getString(R.string.select_document_language));
        radioVietnamese.setText(getString(R.string.vietnamese));
        radioEnglish.setText(getString(R.string.english));
        tvDeleteInstruction.setText(getString(R.string.long_press_to_delete));
    }

    private void setupNavigationDrawer() {
        if (isLoggedIn) {
            guestViewNav.setVisibility(View.GONE);
            historyRecyclerViewNav.setVisibility(View.VISIBLE);
            fetchHistory();
        } else {
            guestViewNav.setVisibility(View.VISIBLE);
            historyRecyclerViewNav.setVisibility(View.GONE);
        }

        btnNewSession.setOnClickListener(v -> {
            startNewOcrSession();
            drawerLayout.closeDrawers();
        });

        btnLoginNav.setOnClickListener(v -> {
            Intent intent = new Intent(MainActivity.this, LoginActivity.class);
            startActivity(intent);
        });
    }

    private void fetchHistory() {
        if (userId == -1) return;

        Request request = new Request.Builder()
                .url(BASE_URL + "/history/" + userId)
                .get()
                .build();

        client.newCall(request).enqueue(new Callback() {
            @Override
            public void onFailure(@NonNull Call call, @NonNull IOException e) {
                // Handle failure (e.g., show a toast)
                runOnUiThread(() -> Toast.makeText(MainActivity.this, "Failed to load history", Toast.LENGTH_SHORT).show());
            }

            @Override
            public void onResponse(@NonNull Call call, @NonNull Response response) throws IOException {
                if (!response.isSuccessful()) {
                    runOnUiThread(() -> Toast.makeText(MainActivity.this, "History fetch failed: " + response.code(), Toast.LENGTH_SHORT).show());
                    return;
                }

                final String responseData = response.body().string();
                try {
                    JSONArray sessionsArray = new JSONArray(responseData);
                    historySessionList.clear();

                    for (int i = 0; i < sessionsArray.length(); i++) {
                        JSONObject sessionObject = sessionsArray.getJSONObject(i);
                        String timestamp = sessionObject.getString("timestamp");
                        int imageCount = sessionObject.getInt("image_count");

                        // --- NEW: Extract image_ids for deletion ---
                        JSONArray imageIdsArray = sessionObject.getJSONArray("image_ids");
                        List<Integer> imageIds = new ArrayList<>();
                        for (int j = 0; j < imageIdsArray.length(); j++) {
                            imageIds.add(imageIdsArray.getInt(j));
                        }
                        // ---

                        JSONArray resultsArray = sessionObject.getJSONArray("results");
                        List<HistoryItemDetail> details = new ArrayList<>();
                        for (int j = 0; j < resultsArray.length(); j++) {
                            JSONObject resultObject = resultsArray.getJSONObject(j);
                            String base64Image = resultObject.getString("image_base64");
                            String text = resultObject.getString("text");
                            Uri imageUri = saveBase64ImageToTempFile(base64Image, "history_" + userId + "_" + i + "_" + j);
                            if (imageUri != null) {
                                details.add(new HistoryItemDetail(imageUri, text));
                            }
                        }
                        historySessionList.add(new HistorySession(timestamp, imageCount, details, imageIds));
                    }

                    runOnUiThread(() -> {
                        historyAdapterNav.setSessions(historySessionList);
                        historyRecyclerViewNav.setVisibility(View.VISIBLE);
                    });

                } catch (JSONException e) {
                    runOnUiThread(() -> Toast.makeText(MainActivity.this, "Failed to parse history", Toast.LENGTH_SHORT).show());
                }
            }
        });
    }

    private Uri saveBase64ImageToTempFile(String base64String, String uniqueId) {
        try {
            byte[] decodedString = Base64.decode(base64String, Base64.DEFAULT);
            File tempFile = File.createTempFile("history_" + uniqueId + "_", ".jpg", getCacheDir());
            FileOutputStream fos = new FileOutputStream(tempFile);
            fos.write(decodedString);
            fos.close();
            return Uri.fromFile(tempFile);
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }

    @Override
    public void onImageClick(Uri imageUri) {
        // When an image in the recycler view is clicked, open it in the viewer for editing
        int position = imageUris.indexOf(imageUri);
        Intent intent = new Intent(this, ImageViewerActivity.class);
        intent.setData(imageUri);
        intent.putExtra("image_position", position);
        imageViewerLauncher.launch(intent);
    }

    @Override
    public void onSessionClick(HistorySession session) {
        // Load session results into the main OCR result view
        ocrResultList.clear();
        for (HistoryItemDetail detail : session.getDetails()) {
            ocrResultList.add(new OcrResult(detail.getImageUri(), detail.getRecognizedText(), false));
        }
        ocrResultAdapter.setOcrResults(ocrResultList);
        drawerLayout.closeDrawers();
    }

    @Override
    public void onDeleteSessionClick(HistorySession session, final int position) {
        new AlertDialog.Builder(this)
            .setTitle(R.string.delete_session_confirmation_title)
            .setMessage(R.string.delete_session_confirmation_message)
            .setPositiveButton(R.string.yes, (dialog, which) -> {
                sendDeleteSessionRequest(session.getImageIds(), position);
            })
            .setNegativeButton(R.string.no, null)
            .show();
    }

    private void sendDeleteSessionRequest(List<Integer> imageIds, final int position) {
        if (imageIds == null || imageIds.isEmpty()) {
            Toast.makeText(this, "Error: No images to delete.", Toast.LENGTH_SHORT).show();
            return;
        }

        JSONObject json = new JSONObject();
        try {
            json.put("image_ids", new JSONArray(imageIds));
        } catch (JSONException e) {
            Toast.makeText(this, "Error creating delete request.", Toast.LENGTH_SHORT).show();
            return;
        }

        RequestBody body = RequestBody.create(json.toString(), MediaType.get("application/json; charset=utf-8"));
        Request request = new Request.Builder()
                .url(BASE_URL + "/history/delete")
                .post(body)
                .build();

        client.newCall(request).enqueue(new Callback() {
            @Override
            public void onFailure(@NonNull Call call, @NonNull IOException e) {
                runOnUiThread(() -> Toast.makeText(MainActivity.this, "Failed to delete session: " + e.getMessage(), Toast.LENGTH_SHORT).show());
            }

            @Override
            public void onResponse(@NonNull Call call, @NonNull Response response) throws IOException {
                if (response.isSuccessful()) {
                    runOnUiThread(() -> {
                        if (position >= 0 && position < historySessionList.size()) {
                            historySessionList.remove(position);
                            historyAdapterNav.notifyItemRemoved(position);
                            historyAdapterNav.notifyItemRangeChanged(position, historySessionList.size());
                            Toast.makeText(MainActivity.this, "Session deleted.", Toast.LENGTH_SHORT).show();
                        }
                    });
                } else {
                    final String errorBody = response.body().string();
                    runOnUiThread(() -> Toast.makeText(MainActivity.this, "Error deleting session: " + errorBody, Toast.LENGTH_LONG).show());
                }
            }
        });
    }

    private void startNewOcrSession() {
        // Clear the selected images
        imageUris.clear();
        imageAdapter.setImageUris(imageUris);

        // Clear the displayed OCR results
        ocrResultList.clear();
        ocrResultAdapter.setOcrResults(ocrResultList);

        // Close the drawer
        drawerLayout.closeDrawers();
    }

    private void adjustNavDrawerForStatusBar() {
        View spacer = findViewById(R.id.status_bar_spacer);
        int statusBarHeight = 0;
        int resourceId = getResources().getIdentifier("status_bar_height", "dimen", "android");
        if (resourceId > 0) {
            statusBarHeight = getResources().getDimensionPixelSize(resourceId);
        }
        LinearLayout.LayoutParams params = (LinearLayout.LayoutParams) spacer.getLayoutParams();
        params.height = statusBarHeight;
        // Set the height of the spacer
        spacer.setLayoutParams(params);
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
        new AlertDialog.Builder(this)
                .setTitle(R.string.stop_ocr_title)
                .setMessage(R.string.stop_ocr_confirmation)
                .setPositiveButton(R.string.yes, (dialog, which) -> {
                    stopOcrRequested = true;
                    if (currentOcrCall != null && !currentOcrCall.isCanceled()) {
                        currentOcrCall.cancel();
                    }
                    updateOcrUiState(false);
                    Toast.makeText(MainActivity.this, R.string.ocr_stopped_message, Toast.LENGTH_SHORT).show();
                })
                .setNegativeButton(R.string.no, null)
                .show();
    }

    private void updateOcrUiState(boolean isRunning) {
        if (isRunning) {
            btnRunOcr.setVisibility(View.GONE);
            btnStopOcr.setVisibility(View.VISIBLE);
            progressBar.setVisibility(View.VISIBLE);
            progressBar.setIndeterminate(false); // We will set progress manually
        } else {
            btnRunOcr.setVisibility(View.VISIBLE);
            btnStopOcr.setVisibility(View.GONE);
            progressBar.setVisibility(View.GONE);
            progressBar.setProgress(0);
            currentOcrCall = null;
        }
    }

    @Override
    public void onImageLongClick(Uri imageUri) {
        new AlertDialog.Builder(this)
                .setTitle(R.string.delete_image_title)
                .setMessage(R.string.delete_image_confirmation)
                .setPositiveButton(R.string.delete, (dialog, which) -> {
                    int position = imageUris.indexOf(imageUri);
                    if (position != -1) {
                        imageUris.remove(position);
                        imageAdapter.notifyItemRemoved(position);
                        imageAdapter.notifyItemRangeChanged(position, imageUris.size());
                    }
                })
                .setNegativeButton(R.string.cancel, null)
                .show();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        try {
            if (ortSession != null) {
                ortSession.close();
            }
            if (detectionSession != null) {
                detectionSession.close();
            }
            if (englishRecognitionSession != null) {
                englishRecognitionSession.close();
            }
            if (ortEnv != null) {
                ortEnv.close();
            }
        } catch (Exception e) {
            Log.e("MainActivity", "Error closing ONNX Runtime sessions: " + e.getMessage());
        }
    }

    private byte[] readBytesFromAsset(String assetFileName) throws IOException {
        InputStream is = getAssets().open(assetFileName);
        ByteArrayOutputStream buffer = new ByteArrayOutputStream();
        int nRead;
        byte[] data = new byte[16384]; // 16KB buffer

        while ((nRead = is.read(data, 0, data.length)) != -1) {
            buffer.write(data, 0, nRead);
        }
        buffer.flush();
        is.close();
        return buffer.toByteArray();
    }
}