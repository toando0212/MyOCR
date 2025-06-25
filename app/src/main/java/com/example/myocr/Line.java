package com.example.myocr;

import org.opencv.core.RotatedRect;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Represents a single line of text, composed of multiple Word objects.
 * This object is immutable.
 */
public class Line {
    private final List<Word> words;
    private final RotatedRect geometry;
    private final String content;

    public Line(List<Word> words, RotatedRect geometry) {
        this.words = words;
        this.geometry = geometry;
        // Pre-compute the full line text upon creation
        this.content = words.stream().map(Word::getValue).collect(Collectors.joining(" "));
    }

    public List<Word> getWords() {
        return words;
    }

    public RotatedRect getGeometry() {
        return geometry;
    }

    /**
     * Gets the full text content of the line.
     * @return A string representing the joined words.
     */
    public String getContent() {
        return content;
    }
} 