package com.example.myocr;

import java.util.HashMap;
import java.util.Map;

public class Vocab {
    public static final int PAD = 0;
    public static final int SOS = 1;
    public static final int EOS = 2;

    public final Map<Integer, String> i2c = new HashMap<>();
    public final Map<String, Integer> c2i = new HashMap<>();
    public int n_words;

    public Vocab(String chars) {
        initVocab(chars);
    }

    public Vocab(String chars, boolean useBlank) {
        initVocab(chars);
        if (useBlank) {
            addBlank();
        }
    }

    private void initVocab(String chars) {
        addWord("<pad>"); // PAD
        addWord("<s>");   // SOS
        addWord("</s>");  // EOS

        for (char c : chars.toCharArray()) {
            addWord(String.valueOf(c));
        }
    }

    private void addWord(String word) {
        if (!c2i.containsKey(word)) {
            c2i.put(word, n_words);
            i2c.put(n_words, word);
            n_words++;
        }
    }

    private void addBlank() {
        addWord(" "); // BLANK
    }

    public String decode(int[] text_int) {
        StringBuilder text = new StringBuilder();
        for (int i : text_int) {
            if (i == SOS) continue;
            if (i == EOS) break;
            if (i == PAD) continue;
            text.append(i2c.get(i));
        }
        return text.toString().replace(" ", " "); // Replace space token with actual space
    }
} 