package com.example.myocr;

import org.opencv.core.RotatedRect;

/**
 * Represents a single recognized word, the most basic element in the OCR structure.
 * This object is immutable.
 */
public class Word {
    private final String value;
    private final double confidence;
    private final RotatedRect geometry;

    public Word(String value, double confidence, RotatedRect geometry) {
        this.value = value;
        this.confidence = confidence;
        this.geometry = geometry;
    }

    public String getValue() {
        return value;
    }

    public double getConfidence() {
        return confidence;
    }

    public RotatedRect getGeometry() {
        return geometry;
    }
} 