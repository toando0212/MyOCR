package com.example.myocr;

import android.graphics.Bitmap;
import android.net.Uri;

public class OcrResult {
    private String text;
    private boolean isProcessing;
    private Uri imageUri;
    private Bitmap previewWithBoxes;

    public OcrResult(Uri imageUri, String text, boolean isProcessing) {
        this.imageUri = imageUri;
        this.text = text;
        this.isProcessing = isProcessing;
        this.previewWithBoxes = null;
    }

    public String getText() {
        return text;
    }

    public void setText(String text) {
        this.text = text;
    }

    public boolean isProcessing() {
        return isProcessing;
    }

    public void setProcessing(boolean processing) {
        isProcessing = processing;
    }

    public Uri getImageUri() {
        return imageUri;
    }

    public void setImageUri(Uri imageUri) {
        this.imageUri = imageUri;
    }

    public Bitmap getPreviewWithBoxes() {
        return previewWithBoxes;
    }

    public void setPreviewWithBoxes(Bitmap previewWithBoxes) {
        this.previewWithBoxes = previewWithBoxes;
    }
} 