package com.example.myocr;

import android.Manifest;
import android.content.ContentValues;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.google.android.material.floatingactionbutton.FloatingActionButton;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;

import okhttp3.*;

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

    private ActivityResultLauncher<Intent> pickImageLauncher;
    private ActivityResultLauncher<Intent> captureImageLauncher;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

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

        // Upload each image to the server
        for (Uri imageUri : imageUris) {
            uploadImageToServer(imageUri, userId);
        }

        // Optionally, you can wait for all uploads to finish before hiding the progress bar
        // For now, just simulate OCR result after a delay
        imageRecyclerView.postDelayed(() -> {
            progressBar.setVisibility(View.GONE);
            tvRecognizedText.setText("[Kết quả nhận diện sẽ hiển thị ở đây]");
        }, 2000);
    }

    private void exportRecognizedText() {
        // TODO: Implement export to PDF or DOCX
        new AlertDialog.Builder(this)
                .setTitle("Xuất file")
                .setMessage("Chức năng xuất ra PDF/DOCX sẽ được bổ sung sau.")
                .setPositiveButton("OK", null)
                .show();
    }

    private void uploadImageToServer(Uri imageUri, int userId) {
        try {
            // Open input stream from URI
            InputStream inputStream = getContentResolver().openInputStream(imageUri);
            if (inputStream == null) {
                runOnUiThread(() -> Toast.makeText(this, "Cannot open image", Toast.LENGTH_SHORT).show());
                return;
            }

            // Create a temp file
            File tempFile = File.createTempFile("upload_", ".jpg", getCacheDir());
            OutputStream outputStream = new FileOutputStream(tempFile);

            // Copy input stream to temp file
            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = inputStream.read(buffer)) != -1) {
                outputStream.write(buffer, 0, bytesRead);
            }
            outputStream.close();
            inputStream.close();

            OkHttpClient client = new OkHttpClient();

            RequestBody fileBody = RequestBody.create(tempFile, MediaType.parse("image/*"));
            MultipartBody requestBody = new MultipartBody.Builder()
                    .setType(MultipartBody.FORM)
                    .addFormDataPart("image", tempFile.getName(), fileBody)
                    .addFormDataPart("user_id", String.valueOf(userId))
                    .build();

            Request request = new Request.Builder()
                    .url("http://192.168.1.219:5000/upload")
                    .post(requestBody)
                    .build();

            client.newCall(request).enqueue(new Callback() {
                @Override
                public void onFailure(Call call, IOException e) {
                    runOnUiThread(() -> Toast.makeText(MainActivity.this, "Upload failed: " + e.getMessage(), Toast.LENGTH_SHORT).show());
                }

                @Override
                public void onResponse(Call call, Response response) throws IOException {
                    if (response.isSuccessful()) {
                        runOnUiThread(() -> Toast.makeText(MainActivity.this, "Upload successful!", Toast.LENGTH_SHORT).show());
                    } else {
                        runOnUiThread(() -> Toast.makeText(MainActivity.this, "Upload failed: " + response.message(), Toast.LENGTH_SHORT).show());
                    }
                }
            });

        } catch (Exception e) {
            runOnUiThread(() -> Toast.makeText(this, "Upload failed: " + e.getMessage(), Toast.LENGTH_SHORT).show());
        }
    }
}