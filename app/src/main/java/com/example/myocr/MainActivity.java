package com.example.myocr;

import android.Manifest;
import android.content.ContentValues;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.Configuration;
import android.database.Cursor;
import android.graphics.Bitmap;
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

public class MainActivity extends AppCompatActivity implements ImageAdapter.OnImageClickListener, HistoryAdapter.OnHistorySessionInteractionListener {
    private static final int REQUEST_CAMERA_PERMISSION = 100;
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

    @Override
    protected void onCreate(Bundle savedInstanceState) {
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
                        } else if (data.getData() != null) {
                            Uri imageUri = data.getData();
                            imageUris.add(imageUri);
                        }
                        imageAdapter.setImageUris(imageUris);
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
                        if (editedImageUri != null) {
                            // Find the position of the original image and replace it with the edited one
                            // For simplicity, let's assume we replace the last edited image.
                            // A more robust solution might involve passing the original image's position.
                            if (!imageUris.isEmpty()) {
                                imageUris.set(imageUris.size() - 1, editedImageUri);
                                imageAdapter.setImageUris(imageUris);
                            }
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
        Toast.makeText(this, R.string.export_coming_soon, Toast.LENGTH_SHORT).show();
    }

    private void performOcrForImage(Uri imageUri, final int position, int userId) {
        byte[] imageData;
        try (InputStream iStream = getContentResolver().openInputStream(imageUri)) {
            if (iStream == null) throw new IOException("Unable to open InputStream.");
            imageData = getBytes(iStream);
        } catch (IOException e) {
            e.printStackTrace();
            runOnUiThread(() -> updateOcrResult(position, getString(R.string.failed_to_process_image, e.getMessage()), false));
            triggerNextImageProcessing();
            return;
        }

        // Dynamically determine the MIME type from the content URI to avoid format errors.
        String mimeType = getContentResolver().getType(imageUri);
        if (mimeType == null) {
            // Fallback for safety, though it should ideally not be null for gallery/camera images
            mimeType = "image/jpeg";
            Log.w("OCR_MIME_TYPE", "MIME type was null for URI: " + imageUri + ". Defaulting to image/jpeg.");
        }
        MediaType mediaType = MediaType.parse(mimeType);

        RequestBody requestBody = new MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("image", "image.jpg", RequestBody.create(imageData, mediaType))
                .addFormDataPart("language", radioVietnamese.isChecked() ? "vie" : "eng")
                .addFormDataPart("user_id", String.valueOf(userId))
                .build();

        Request request = new Request.Builder()
                .url(BASE_URL + "/classify")
                .post(requestBody)
                .build();

        currentOcrCall = client.newCall(request);
        currentOcrCall.enqueue(new Callback() {
            @Override
            public void onFailure(@NonNull Call call, @NonNull IOException e) {
                if (call.isCanceled()) {
                    Log.d("OCR_STOP", "Call was canceled by user.");
                    // UI is reset by the stop button listener, no need to trigger next
                } else {
                    runOnUiThread(() -> {
                        updateOcrResult(position, "Network Error", false);
                        Toast.makeText(MainActivity.this, getString(R.string.ocr_request_failed, e.getMessage()), Toast.LENGTH_SHORT).show();
                    });
                    triggerNextImageProcessing(); // Try next image on failure
                }
            }

            @Override
            public void onResponse(@NonNull Call call, @NonNull Response response) throws IOException {
                try (ResponseBody responseBody = response.body()) {
                    if (!response.isSuccessful()) {
                        final String errorBody = responseBody != null ? responseBody.string() : "Unknown error";
                        runOnUiThread(() -> {
                            updateOcrResult(position, "Error: " + response.code(), false);
                            Toast.makeText(MainActivity.this, "OCR failed: " + errorBody, Toast.LENGTH_SHORT).show();
                        });
                        return; // Do not trigger next if there's a server error for this image
                    }

                    final String responseData = responseBody != null ? responseBody.string() : "";
                    try {
                        JSONObject jsonObject = new JSONObject(responseData);
                        JSONArray results = jsonObject.getJSONArray("results");
                        if (results.length() > 0) {
                            String recognizedText = results.getJSONObject(0).getString("text");
                            runOnUiThread(() -> updateOcrResult(position, recognizedText, false));
                        } else {
                            runOnUiThread(() -> updateOcrResult(position, getString(R.string.no_text_recognized), false));
                        }
                    } catch (JSONException e) {
                        runOnUiThread(() -> {
                            updateOcrResult(position, "Parse Error", false);
                            Toast.makeText(MainActivity.this, R.string.failed_to_parse_ocr_result, Toast.LENGTH_SHORT).show();
                        });
                    }
                } finally {
                    triggerNextImageProcessing();
                }
            }
        });
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
        // This is for the top image list, not history
        // OLD: show delete confirmation dialog
        // NEW: Open image in a viewer
        Intent intent = new Intent(Intent.ACTION_VIEW);
        intent.setDataAndType(imageUri, "image/*");
        intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION); // Important for content URIs
        try {
            startActivity(intent);
        } catch (Exception e) {
            Toast.makeText(this, "No application can handle this request. Please install a gallery app.", Toast.LENGTH_SHORT).show();
            e.printStackTrace();
        }
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
}