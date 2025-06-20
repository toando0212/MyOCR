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

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;

import okhttp3.*;
import org.json.JSONArray;
import org.json.JSONObject;

public class MainActivity extends AppCompatActivity {
    private static final int REQUEST_CAMERA_PERMISSION = 100;
    private RecyclerView imageRecyclerView;
    private ImageAdapter imageAdapter;
    private List<Uri> imageUris = new ArrayList<>();
    private Button btnRunOcr, btnExport;
    private ProgressBar progressBar;
    private TextView tvRecognizedText;
    private FloatingActionButton fab;
    private Uri cameraImageUri;
    private RadioButton radioVietnamese, radioEnglish;

    private ActivityResultLauncher<Intent> pickImageLauncher;
    private ActivityResultLauncher<Intent> captureImageLauncher;

    private DrawerLayout drawerLayout;
    private ActionBarDrawerToggle toggle;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Toolbar toolbar = findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);

        imageRecyclerView = findViewById(R.id.imageRecyclerView);
        btnRunOcr = findViewById(R.id.btnRunOcr);
        btnExport = findViewById(R.id.btnExport);
        progressBar = findViewById(R.id.progressBar);
        tvRecognizedText = findViewById(R.id.tvRecognizedText);
        fab = findViewById(R.id.fab);

        imageAdapter = new ImageAdapter(this, imageUris);
        imageRecyclerView.setLayoutManager(new LinearLayoutManager(this, LinearLayoutManager.HORIZONTAL, false));
        imageRecyclerView.setAdapter(imageAdapter);

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

        fab.setOnClickListener(v -> showImageSourceDialog());

        btnRunOcr.setOnClickListener(v -> runOcrOnImages());
        btnExport.setOnClickListener(v -> exportRecognizedText());

        radioVietnamese = findViewById(R.id.radioVietnamese);
        radioEnglish = findViewById(R.id.radioEnglish);

        // Set progress bar to indeterminate mode
        progressBar.setIndeterminate(true);
        progressBar.setVisibility(View.GONE);

        drawerLayout = findViewById(R.id.drawer_layout);
        NavigationView navigationView = findViewById(R.id.nav_view);

        toggle = new ActionBarDrawerToggle(this, drawerLayout, toolbar, R.string.navigation_drawer_open, R.string.navigation_drawer_close);
        drawerLayout.addDrawerListener(toggle);

        getSupportActionBar().setDisplayHomeAsUpEnabled(true);

        navigationView.setNavigationItemSelectedListener(item -> {
            int id = item.getItemId();
            if (id == R.id.nav_history) {
                // Handle history action
                Toast.makeText(MainActivity.this, "History selected", Toast.LENGTH_SHORT).show();
            }
            drawerLayout.closeDrawers();
            return true;
        });
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
        String[] options = {"Chụp ảnh", "Tải lên từ thiết bị"};
        new AlertDialog.Builder(this)
                .setTitle("Chọn nguồn ảnh")
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
            tvRecognizedText.setText("Vui lòng chọn ảnh trước.");
            return;
        }
        progressBar.setVisibility(View.VISIBLE);
        tvRecognizedText.setText("");

        int userId = 1; // Replace with actual user ID logic

        // Process each image sequentially and append results
        // Use a StringBuilder to accumulate text from all images
        StringBuilder allRecognizedText = new StringBuilder();
        int totalImages = imageUris.size();
        int[] imagesProcessed = {0}; // Use an array to be modifiable inside lambda

        for (Uri imageUri : imageUris) {
            performOcrForImage(imageUri, userId, allRecognizedText, totalImages, imagesProcessed);
        }
    }

    private void exportRecognizedText() {
        // TODO: Implement export to PDF or DOCX
        new AlertDialog.Builder(this)
                .setTitle("Xuất file")
                .setMessage("Chức năng xuất ra PDF/DOCX sẽ được bổ sung sau.")
                .setPositiveButton("OK", null)
                .show();
    }

    private void performOcrForImage(Uri imageUri, int userId, StringBuilder allRecognizedText, int totalImages, int[] imagesProcessed) {
        try {
            InputStream inputStream = getContentResolver().openInputStream(imageUri);
            if (inputStream == null) {
                runOnUiThread(() -> Toast.makeText(this, "Cannot open image", Toast.LENGTH_SHORT).show());
                return;
            }

            File tempFile = File.createTempFile("upload_", ".jpg", getCacheDir());
            try (OutputStream outputStream = new FileOutputStream(tempFile)) {
                byte[] buffer = new byte[4096];
                int bytesRead;
                while ((bytesRead = inputStream.read(buffer)) != -1) {
                    outputStream.write(buffer, 0, bytesRead);
                }
            } finally {
                inputStream.close();
            }

            OkHttpClient client = new OkHttpClient.Builder()
                    .readTimeout(5, java.util.concurrent.TimeUnit.MINUTES)
                    .connectTimeout(1, java.util.concurrent.TimeUnit.MINUTES) // Add connect timeout
                    .writeTimeout(1, java.util.concurrent.TimeUnit.MINUTES) // Add write timeout
                    .build();

            RequestBody fileBody = RequestBody.create(tempFile, MediaType.parse("image/*"));
            String language = radioVietnamese.isChecked() ? "vie" : "eng";

            MultipartBody requestBody = new MultipartBody.Builder()
                    .setType(MultipartBody.FORM)
                    .addFormDataPart("image", tempFile.getName(), fileBody)
                    .addFormDataPart("language", language)
                    .addFormDataPart("user_id", String.valueOf(userId))
                    .build();

            // IMPORTANT: Use your actual server URL here
            Request request = new Request.Builder()
                    .url("https://3e8f-42-114-227-240.ngrok-free.app/classify")
                    .post(requestBody)
                    .build();

            client.newCall(request).enqueue(new Callback() {
                @Override
                public void onFailure(@NonNull Call call, @NonNull IOException e) {
                    runOnUiThread(() -> {
                        Toast.makeText(MainActivity.this, "OCR Request failed: " + e.getMessage(), Toast.LENGTH_LONG).show();
                        // Check if all images have been processed (even on failure)
                        imagesProcessed[0]++;
                        if (imagesProcessed[0] == totalImages) {
                            progressBar.setVisibility(View.GONE);
                        }
                    });
                }

                @Override
                public void onResponse(@NonNull Call call, @NonNull Response response) throws IOException {
                    final String responseBody = response.body().string();
                    if (response.isSuccessful()) {
                        try {
                            JSONObject json = new JSONObject(responseBody);
                            JSONArray results = json.getJSONArray("results");
                            StringBuilder currentImageText = new StringBuilder();
                            for (int i = 0; i < results.length(); i++) {
                                JSONObject block = results.getJSONObject(i);
                                currentImageText.append(block.getString("text")).append("\n");
                            }
                            // Append the result of the current image to the total
                            allRecognizedText.append(currentImageText);

                        } catch (Exception e) {
                            runOnUiThread(() -> Toast.makeText(MainActivity.this, "Failed to parse OCR result", Toast.LENGTH_SHORT).show());
                        }
                    } else {
                         runOnUiThread(() -> {
                            try {
                                // Try to parse error message from server
                                JSONObject json = new JSONObject(responseBody);
                                String error = json.optString("error", response.message());
                                Toast.makeText(MainActivity.this, "OCR failed: " + error, Toast.LENGTH_LONG).show();
                            } catch (Exception e) {
                                Toast.makeText(MainActivity.this, "OCR failed: " + response.message(), Toast.LENGTH_LONG).show();
                            }
                        });
                    }

                    // This block runs regardless of success or failure
                    runOnUiThread(() -> {
                        imagesProcessed[0]++;
                        // If this is the last image, update the TextView and hide the progress bar
                        if (imagesProcessed[0] == totalImages) {
                            progressBar.setVisibility(View.GONE);
                            tvRecognizedText.setText(allRecognizedText.toString());
                            if(allRecognizedText.length() == 0) {
                                tvRecognizedText.setText("No text recognized.");
                            }
                        }
                    });
                }
            });

        } catch (Exception e) {
            runOnUiThread(() -> {
                Toast.makeText(this, "Failed to process image: " + e.getMessage(), Toast.LENGTH_SHORT).show();
                progressBar.setVisibility(View.GONE);
            });
        }
    }
}