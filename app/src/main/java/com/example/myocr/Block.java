package com.example.myocr;

import org.opencv.core.RotatedRect;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Represents a block of text, composed of multiple Line objects.
 * This typically corresponds to a paragraph.
 * This object is immutable.
 */
public class Block {
    private final List<Line> lines;
    private final RotatedRect geometry;
    private final String content;

    public Block(List<Line> lines, RotatedRect geometry) {
        this.lines = lines;
        this.geometry = geometry;
        // Pre-compute the full block text upon creation
        this.content = lines.stream().map(Line::getContent).collect(Collectors.joining("\n"));
    }

    public List<Line> getLines() {
        return lines;
    }

    public RotatedRect getGeometry() {
        return geometry;
    }

    /**
     * Gets the full text content of the block, with lines separated by newlines.
     * @return A string representing the joined lines.
     */
    public String getContent() {
        return content;
    }
} 