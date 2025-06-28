package com.example.myocr;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Vocab {
    // --- THUỘC TÍNH THUẦN CTC ---
    public final Map<Integer, String> i2c = new HashMap<>();
    public final Map<String, Integer> c2i = new HashMap<>();
    private final int blankIndex;

    public Vocab(String chars) {
        // 1. Chỉ thêm các ký tự từ chuỗi vocab được cung cấp
        for (char c : chars.toCharArray()) {
            addWord(String.valueOf(c));
        }
        
        // 2. Index của blank token sẽ là index ngay sau ký tự cuối cùng.
        // Ví dụ: nếu có 200 ký tự (index 0-199), blankIndex sẽ là 200.
        blankIndex = c2i.size();
    }

    private void addWord(String word) {
        if (!c2i.containsKey(word)) {
            int index = c2i.size();
            c2i.put(word, index);
            i2c.put(index, word);
        }
    }

    /**
     * Decodes the raw output logits from a CTC-based model.
     * This method implements a CTC greedy decoder.
     * @param logits The output from the ONNX model, shape: [1, sequence_length, vocab_size]
     * @return The decoded string.
     */
    public String decode(float[][][] logits) {
        // 1. Get the best path (most likely character at each time step)
        List<Integer> bestPath = new ArrayList<>();
        float[][] sequence = logits[0]; // Shape: [sequence_length, vocab_size]

        for (float[] step : sequence) {
            // Find the character with the highest probability (argmax)
            int bestCharIndex = -1;
            float maxLogit = Float.NEGATIVE_INFINITY;
            for (int i = 0; i < step.length; i++) {
                if (step[i] > maxLogit) {
                    maxLogit = step[i];
                    bestCharIndex = i;
                }
            }
            bestPath.add(bestCharIndex);
        }

        // 2. Decode the path using CTC rules
        List<Integer> collapsedPath = new ArrayList<>();
        int lastIndex = -1;
        for (int currentIndex : bestPath) {
            // Rule 1: Collapse repeated characters.
            // If the current index is the same as the last one, skip it.
            if (currentIndex == lastIndex) {
                continue;
            }
            // Rule 2: Ignore the blank token.
            if (currentIndex != this.blankIndex) {
                collapsedPath.add(currentIndex);
            }
            // Update the last index to the current one for the next iteration's check.
            lastIndex = currentIndex;
        }

        // 3. Convert the collapsed path of indices to a final string
        StringBuilder sb = new StringBuilder();
        for (int index : collapsedPath) {
            String character = i2c.get(index);
            if (character != null) {
                sb.append(character);
            }
        }
        return sb.toString();
    }

    // Hàm này không còn cần thiết vì model đã được build với vocab cố định,
    // nhưng giữ lại cũng không sao.
    public int getVocabSize() {
        // Kích thước output của model CTC = số ký tự + 1 (cho blank token)
        return c2i.size() + 1;
    }
} 