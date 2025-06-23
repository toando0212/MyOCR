package com.example.myocr;

import android.net.Uri;

public class OcrResult {
    private final Uri imageUri;
    private String recognizedText;
    private boolean isProcessing;

    public OcrResult(Uri imageUri, String recognizedText, boolean isProcessing) {
        this.imageUri = imageUri;
        this.recognizedText = recognizedText;
        this.isProcessing = isProcessing;
    }

    public Uri getImageUri() {
        return imageUri;
    }

    public String getRecognizedText() {
        return recognizedText;
    }

    public void setRecognizedText(String recognizedText) {
        this.recognizedText = recognizedText;
    }

    public boolean isProcessing() {
        return isProcessing;
    }

    public void setProcessing(boolean processing) {
        isProcessing = processing;
    }
} 