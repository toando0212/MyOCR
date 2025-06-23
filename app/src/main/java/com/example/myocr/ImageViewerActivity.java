package com.example.myocr;

import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import com.yalantis.ucrop.UCrop;

import java.io.File;

public class ImageViewerActivity extends AppCompatActivity {

    private Uri imageUri;
    private ImageView fullScreenImageView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_image_viewer);

        fullScreenImageView = findViewById(R.id.fullScreenImageView);
        Button btnEdit = findViewById(R.id.btnEdit);

        imageUri = getIntent().getData();
        if (imageUri != null) {
            fullScreenImageView.setImageURI(imageUri);
        }

        btnEdit.setOnClickListener(v -> startImageEditor(imageUri));

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
                    imageUri = resultUri;
                    fullScreenImageView.setImageURI(imageUri);
                    Toast.makeText(this, "Image cropped/rotated successfully!", Toast.LENGTH_SHORT).show();
                    Intent resultIntent = new Intent();
                    resultIntent.setData(resultUri);
                    setResult(RESULT_OK, resultIntent);
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
} 