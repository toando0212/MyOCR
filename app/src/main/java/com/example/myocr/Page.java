package com.example.myocr;

import android.graphics.Bitmap;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Represents a single page of a document, composed of multiple Block objects.
 * This is the top-level container for the OCR result of a single image.
 * This object is immutable.
 */
public class Page {
    private final List<Block> blocks;
    private final int pageIndex;
    private final int width;
    private final int height;
    private final String content;
    private final Bitmap previewImage; // Optional: can hold the image with drawn bounding boxes

    public Page(List<Block> blocks, int pageIndex, int width, int height, Bitmap previewImage) {
        this.blocks = blocks;
        this.pageIndex = pageIndex;
        this.width = width;
        this.height = height;
        this.previewImage = previewImage;
        // Pre-compute the full page text upon creation
        this.content = blocks.stream().map(Block::getContent).collect(Collectors.joining("\n\n"));
    }

    // Constructor mới cho phép truyền content tuỳ ý
    public Page(List<Block> blocks, int pageIndex, int width, int height, Bitmap previewImage, String content) {
        this.blocks = blocks;
        this.pageIndex = pageIndex;
        this.width = width;
        this.height = height;
        this.previewImage = previewImage;
        this.content = content;
    }

    public List<Block> getBlocks() {
        return blocks;
    }

    public int getPageIndex() {
        return pageIndex;
    }

    public int getWidth() {
        return width;
    }

    public int getHeight() {
        return height;
    }

    public Bitmap getPreviewImage() {
        return previewImage;
    }

    /**
     * Gets the full text content of the page, with blocks separated by double newlines.
     * @return A string representing the entire page's text.
     */
    public String getContent() {
        return content;
    }

    /**
     * Returns a new Page with the same properties but updated content.
     */
    public Page updateContent(String newContent) {
        return new Page(blocks, pageIndex, width, height, previewImage, newContent);
    }
} 