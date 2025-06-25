package com.example.myocr;

import android.net.Uri;

/**
 * A state holder for an item in the OCR results RecyclerView.
 * It holds the original image URI and the resulting Page object after OCR.
 */
public class OcrResult {
    private final Uri imageUri;
    private Page page; // This will hold the entire structured result
    private boolean isProcessing;
    private String error; // To hold any error messages
    private String recognizedText; // To hold simple text, especially from history

    public OcrResult(Uri imageUri, boolean isProcessing) {
        this.imageUri = imageUri;
        this.isProcessing = isProcessing;
        this.page = null;
        this.error = null;
    }

    // New constructor for when we have pre-existing text (e.g., from history)
    public OcrResult(Uri imageUri, String recognizedText, boolean isProcessing) {
        this.imageUri = imageUri;
        this.recognizedText = recognizedText;
        this.isProcessing = isProcessing;
        this.page = null; // No page object initially when loading from history text
        this.error = null;
    }

    public Uri getImageUri() {
        return imageUri;
    }

    public Page getPage() {
        return page;
    }

    public void setPage(Page page) {
        this.page = page;
    }

    public boolean isProcessing() {
        return isProcessing;
    }

    public void setProcessing(boolean processing) {
        isProcessing = processing;
    }

    public String getError() {
        return error;
    }

    public void setError(String error) {
        this.error = error;
    }

    /**
     * Gets the recognized text.
     * Prefers the detailed content from the Page object if available,
     * otherwise returns the simpler text (e.g., from history).
     * @return The OCR text.
     */
    public String getText() {
        if (page != null && page.getContent() != null && !page.getContent().isEmpty()) {
            return page.getContent();
        }
        return recognizedText != null ? recognizedText : "";
    }
} 