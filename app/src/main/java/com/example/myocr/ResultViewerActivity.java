package com.example.myocr;

import android.net.Uri;
import android.os.Bundle;
import androidx.appcompat.app.AppCompatActivity;
import com.github.chrisbanes.photoview.PhotoView;

public class ResultViewerActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_result_viewer);

        PhotoView photoView = findViewById(R.id.photo_view);
        String imageUriString = getIntent().getStringExtra("image_uri");

        if (imageUriString != null) {
            Uri imageUri = Uri.parse(imageUriString);
            photoView.setImageURI(imageUri);
        }
    }
} 