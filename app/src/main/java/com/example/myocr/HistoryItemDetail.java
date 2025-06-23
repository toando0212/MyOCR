package com.example.myocr;

import android.net.Uri;

public class HistoryItemDetail {
    private final Uri imageUri;
    private final String recognizedText;

    public HistoryItemDetail(Uri imageUri, String recognizedText) {
        this.imageUri = imageUri;
        this.recognizedText = recognizedText;
    }

    public Uri getImageUri() {
        return imageUri;
    }

    public String getRecognizedText() {
        return recognizedText;
    }
} 