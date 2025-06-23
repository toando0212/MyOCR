package com.example.myocr;

import android.net.Uri;
import java.util.List;
import java.util.stream.Collectors;

public class HistorySession {
    private final String timestamp;
    private final int imageCount;
    private final List<HistoryItemDetail> details;
    private final List<Integer> imageIds;

    public HistorySession(String timestamp, int imageCount, List<HistoryItemDetail> details, List<Integer> imageIds) {
        this.timestamp = timestamp;
        this.imageCount = imageCount;
        this.details = details;
        this.imageIds = imageIds;
    }

    public String getTimestamp() {
        return timestamp;
    }

    public int getImageCount() {
        return imageCount;
    }

    public List<HistoryItemDetail> getDetails() {
        return details;
    }

    public List<Integer> getImageIds() {
        return imageIds;
    }

    // Helper method for the adapter to get the preview image
    public Uri getPreviewImageUri() {
        if (details != null && !details.isEmpty()) {
            return details.get(0).getImageUri();
        }
        return null;
    }

    // Helper method for the adapter to get the combined text for preview
    public String getFullTextPreview() {
        if (details == null || details.isEmpty()) {
            return "";
        }
        return details.stream()
                      .map(HistoryItemDetail::getRecognizedText)
                      .collect(Collectors.joining("\n"));
    }
} 