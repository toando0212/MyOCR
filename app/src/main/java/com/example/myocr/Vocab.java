package com.example.myocr;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Vocab {
    // These values MUST match the token IDs used during the model's training in Python
    // SOS, EOS, PAD tokens are added AFTER the main vocabulary
    public final int SOS;
    public final int EOS;
    public final int PAD;

    public final Map<Integer, String> i2c = new HashMap<>();
    public final Map<String, Integer> c2i = new HashMap<>();
    private int n_words = 0;
    private final int blankIndex;

    public Vocab(String chars) {
        // For CTC models like ViTSTR, the vocabulary consists only of the printable characters.
        // The special "blank" token is handled implicitly by the model and decoder.
        for (char c : chars.toCharArray()) {
            addWord(String.valueOf(c));
        }
        // The blank token is assumed to have an index equal to the number of characters.
        // e.g., if there are 95 chars (0-94), blank is 95.
        blankIndex = c2i.size();

        // 2. Add special tokens AFTER the character vocabulary.
        // Their indices must match the Python implementation (len(vocab), len(vocab)+1, ...)
        SOS = n_words;
        addWord("<s>");   // SOS will have the index of len(chars)

        EOS = n_words;
        addWord("</s>");  // EOS will have the index of len(chars) + 1

        PAD = n_words;
        addWord("<pad>");  // PAD will have the index of len(chars) + 2
    }

    private void addWord(String word) {
        if (!c2i.containsKey(word)) {
            int index = c2i.size();
            c2i.put(word, index);
            i2c.put(index, word);
            n_words++;
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

    public int getVocabSize() {
        // The model's vocabulary size is the number of characters + 1 for the blank token.
        return i2c.size() + 1;
    }
} 