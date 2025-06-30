package com.example.myocr;

import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.core.content.ContextCompat;
import androidx.recyclerview.widget.RecyclerView;

import com.bumptech.glide.Glide;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.List;
import java.util.Locale;

import org.languagetool.JLanguageTool;
import org.languagetool.language.AmericanEnglish;
import org.languagetool.rules.RuleMatch;
import android.util.Log;
import okhttp3.*;
import org.json.JSONArray;
import org.json.JSONObject;
import android.content.res.AssetManager;
import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.HashSet;
import android.app.AlertDialog;
import android.widget.ScrollView;
import android.widget.EditText;
import android.text.SpannableString;
import android.text.Spanned;
import android.text.style.ForegroundColorSpan;
import android.text.style.ClickableSpan;
import android.view.MotionEvent;
import android.text.method.LinkMovementMethod;

public class OcrResultAdapter extends RecyclerView.Adapter<OcrResultAdapter.ViewHolder> {

    private final Context context;
    private List<OcrResult> ocrResultList;
    private boolean isEnglishMode = false;
    private static HashSet<String> vietnameseDict;
    private static boolean dictLoaded = false;
    private final OkHttpClient client;

    public OcrResultAdapter(Context context, List<OcrResult> ocrResultList, boolean isEnglishMode, OkHttpClient client) {
        this.context = context;
        this.ocrResultList = ocrResultList;
        this.isEnglishMode = isEnglishMode;
        this.client = client;
    }

    public void setOcrResults(List<OcrResult> ocrResults) {
        this.ocrResultList = ocrResults;
        notifyDataSetChanged();
    }

    @NonNull
    @Override
    public ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(context).inflate(R.layout.item_ocr_result, parent, false);
        return new ViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull ViewHolder holder, int position) {
        OcrResult result = ocrResultList.get(position);

        if (result.isProcessing()) {
            holder.progressBar.setVisibility(View.VISIBLE);
            holder.tvOcrResult.setVisibility(View.GONE);
            holder.ivPreview.setVisibility(View.GONE);
        } else {
            holder.progressBar.setVisibility(View.GONE);
            holder.tvOcrResult.setVisibility(View.VISIBLE);
            
            if (result.getError() != null) {
                // Handle error case
                holder.tvOcrResult.setText(result.getError());
                holder.tvOcrResult.setTextColor(ContextCompat.getColor(context, android.R.color.holo_red_dark));
                holder.ivPreview.setVisibility(View.GONE);

            } else if (result.getPage() != null) {
                // Handle success case with a valid Page object
                Page page = result.getPage();
                holder.tvOcrResult.setText(page.getContent());
                holder.tvOcrResult.setTextColor(ContextCompat.getColor(context, android.R.color.black)); // Reset color
                
                Bitmap previewBitmap = page.getPreviewImage();
                if (previewBitmap != null) {
                    holder.ivPreview.setVisibility(View.VISIBLE);
                    holder.ivPreview.setImageBitmap(previewBitmap);
                    holder.ivPreview.setOnClickListener(v -> {
                        // Save bitmap to a temp file and pass the URI to the new activity
                        try {
                            File cachePath = new File(context.getCacheDir(), "images");
                            cachePath.mkdirs();
                            File tempFile = new File(cachePath, "image_preview_" + position + ".png");
                            FileOutputStream stream = new FileOutputStream(tempFile);
                            previewBitmap.compress(Bitmap.CompressFormat.PNG, 100, stream);
                            stream.close();

                            Intent intent = new Intent(context, ResultViewerActivity.class);
                            intent.putExtra("image_uri", Uri.fromFile(tempFile).toString());
                            intent.putExtra("ocr_text", page.getContent());
                            context.startActivity(intent);

                        } catch (IOException e) {
                            e.printStackTrace();
                        }
                    });
                } else {
                    holder.ivPreview.setVisibility(View.GONE);
                }
            } else {
                // Handle case where processing is done but page is null and no error
                holder.tvOcrResult.setText("No result available.");
                holder.ivPreview.setVisibility(View.GONE);
            }
        }

        Button btnSpellCheck = holder.btnSpellCheck;
        TextView tvSpellLang = holder.itemView.findViewById(R.id.tvSpellLang);
        Log.d("SpellCheckDebug", "position=" + position + ", isProcessing=" + result.isProcessing() + ", error=" + result.getError() + ", isEnglishMode=" + isEnglishMode + ", text='" + result.getText() + "'");
        if (!result.isProcessing() && result.getError() == null && result.getText() != null && !result.getText().isEmpty()) {
            btnSpellCheck.setVisibility(View.VISIBLE);
            if (isEnglishMode) {
                if (tvSpellLang != null) {
                    tvSpellLang.setVisibility(View.VISIBLE);
                    tvSpellLang.setText("English");
                }
                btnSpellCheck.setOnClickListener(v -> {
                    btnSpellCheck.setEnabled(false);
                    String originalText = result.getText();
                    JSONObject json = new JSONObject();
                    try {
                        json.put("text", originalText);
                        json.put("language", "en_US");
                    } catch (Exception e) {
                        btnSpellCheck.setEnabled(true);
                        Toast.makeText(context, "Lỗi tạo request: " + e.getMessage(), Toast.LENGTH_LONG).show();
                        return;
                    }
                    String spellcheckUrl = MainActivity.BASE_URL + "/spellcheck";
                    RequestBody body = RequestBody.create(json.toString(), MediaType.get("application/json; charset=utf-8"));
                    Request request = new Request.Builder()
                            .url(spellcheckUrl)
                            .post(body)
                            .build();
                    client.newCall(request).enqueue(new Callback() {
                        @Override
                        public void onFailure(Call call, IOException e) {
                            Log.e("SpellCheckDebug", "onFailure: " + e.getMessage());
                            ((android.app.Activity) context).runOnUiThread(() -> {
                                Toast.makeText(context, "Cần internet để sửa lỗi chính tả", Toast.LENGTH_LONG).show();
                                btnSpellCheck.setEnabled(true);
                            });
                        }
                        @Override
                        public void onResponse(Call call, Response response) throws IOException {
                            try {
                                Log.d("SpellCheckDebug", "onResponse START. Successful: " + response.isSuccessful());

                                if (!response.isSuccessful()) {
                                    Log.e("SpellCheckDebug", "Response not successful. Code: " + response.code());
                                    ((android.app.Activity) context).runOnUiThread(() -> {
                                        Toast.makeText(context, "Lỗi server: " + response.message(), Toast.LENGTH_LONG).show();
                                        btnSpellCheck.setEnabled(true);
                                    });
                                    return;
                                }

                                final String responseBody = response.body().string();
                                Log.d("SpellCheckDebug", "Successfully read response body. Length: " + responseBody.length());

                                final JSONObject json = new JSONObject(responseBody);
                                final String fixedText = json.optString("corrected_text", originalText);
                                final JSONArray typosArray = json.optJSONArray("typos");
                                Log.d("SpellCheckDebug", "Parsed JSON, got fixedText. Length: " + fixedText.length());

                                ((android.app.Activity) context).runOnUiThread(() -> {
                                    try {
                                        Log.d("SpellCheckDebug", "Inside runOnUiThread. Preparing to show dialog.");
                                        Toast.makeText(context, "Đã nhận kết quả, đang hiển thị...", Toast.LENGTH_SHORT).show();
                                        showSpellCheckDialogWithTypos(originalText, fixedText, typosArray, result, position, btnSpellCheck);
                                        Log.d("SpellCheckDebug", "showSpellCheckDialog call completed.");
                                    } catch (Exception e) {
                                        Log.e("SpellCheckDebug", "ERROR inside runOnUiThread (showing dialog)", e);
                                        Toast.makeText(context, "Lỗi hiển thị kết quả: " + e.getMessage(), Toast.LENGTH_LONG).show();
                                        btnSpellCheck.setEnabled(true);
                                    }
                                });

                            } catch (Exception e) {
                                Log.e("SpellCheckDebug", "FATAL ERROR in onResponse background task", e);
                                ((android.app.Activity) context).runOnUiThread(() -> {
                                    Toast.makeText(context, "Lỗi xử lý phản hồi: " + e.getMessage(), Toast.LENGTH_LONG).show();
                                    btnSpellCheck.setEnabled(true);
                                });
                            }
                        }
                    });
                });
            } else { // Vietnamese mode
                if (tvSpellLang != null) {
                    tvSpellLang.setVisibility(View.VISIBLE);
                    tvSpellLang.setText("Vietnamese");
                }
                btnSpellCheck.setOnClickListener(v -> {
                    btnSpellCheck.setEnabled(false);
                    String originalText = result.getText();
                    JSONObject json = new JSONObject();
                    try {
                        json.put("text", originalText);
                        json.put("language", "vi_VN");
                    } catch (Exception e) {
                        btnSpellCheck.setEnabled(true);
                        Toast.makeText(context, "Lỗi tạo request: " + e.getMessage(), Toast.LENGTH_LONG).show();
                        return;
                    }
                    String spellcheckUrl = MainActivity.BASE_URL + "/spellcheck";
                    RequestBody body = RequestBody.create(json.toString(), MediaType.get("application/json; charset=utf-8"));
                    Request request = new Request.Builder()
                            .url(spellcheckUrl)
                            .post(body)
                            .build();
                    client.newCall(request).enqueue(new Callback() {
                        @Override
                        public void onFailure(Call call, IOException e) {
                            Log.e("SpellCheckDebug", "onFailure: " + e.getMessage());
                            ((android.app.Activity) context).runOnUiThread(() -> {
                                Toast.makeText(context, "Cần internet để sửa lỗi chính tả", Toast.LENGTH_LONG).show();
                                btnSpellCheck.setEnabled(true);
                            });
                        }
                        @Override
                        public void onResponse(Call call, Response response) throws IOException {
                            try {
                                Log.d("SpellCheckDebug", "onResponse START. Successful: " + response.isSuccessful());

                                if (!response.isSuccessful()) {
                                    Log.e("SpellCheckDebug", "Response not successful. Code: " + response.code());
                                    ((android.app.Activity) context).runOnUiThread(() -> {
                                        Toast.makeText(context, "Lỗi server: " + response.message(), Toast.LENGTH_LONG).show();
                                        btnSpellCheck.setEnabled(true);
                                    });
                                    return;
                                }

                                final String responseBody = response.body().string();
                                Log.d("SpellCheckDebug", "Successfully read response body. Length: " + responseBody.length());

                                final JSONObject json = new JSONObject(responseBody);
                                final String fixedText = json.optString("corrected_text", originalText);
                                final JSONArray typosArray = json.optJSONArray("typos");
                                Log.d("SpellCheckDebug", "Parsed JSON, got fixedText. Length: " + fixedText.length());

                                ((android.app.Activity) context).runOnUiThread(() -> {
                                    try {
                                        Log.d("SpellCheckDebug", "Inside runOnUiThread. Preparing to show dialog.");
                                        Toast.makeText(context, "Đã nhận kết quả, đang hiển thị...", Toast.LENGTH_SHORT).show();
                                        showSpellCheckDialogWithTypos(originalText, fixedText, typosArray, result, position, btnSpellCheck);
                                        Log.d("SpellCheckDebug", "showSpellCheckDialog call completed.");
                                    } catch (Exception e) {
                                        Log.e("SpellCheckDebug", "ERROR inside runOnUiThread (showing dialog)", e);
                                        Toast.makeText(context, "Lỗi hiển thị kết quả: " + e.getMessage(), Toast.LENGTH_LONG).show();
                                        btnSpellCheck.setEnabled(true);
                                    }
                                });

                            } catch (Exception e) {
                                Log.e("SpellCheckDebug", "FATAL ERROR in onResponse background task", e);
                                ((android.app.Activity) context).runOnUiThread(() -> {
                                    Toast.makeText(context, "Lỗi xử lý phản hồi: " + e.getMessage(), Toast.LENGTH_LONG).show();
                                    btnSpellCheck.setEnabled(true);
                                });
                            }
                        }
                    });
                });
            }
        } else {
            btnSpellCheck.setVisibility(View.GONE);
            btnSpellCheck.setOnClickListener(null);
            if (tvSpellLang != null) tvSpellLang.setVisibility(View.GONE);
        }
    }

    @Override
    public int getItemCount() {
        return ocrResultList.size();
    }

    public static class ViewHolder extends RecyclerView.ViewHolder {
        TextView tvOcrResult;
        ProgressBar progressBar;
        ImageView ivPreview;
        Button btnSpellCheck;

        public ViewHolder(View view) {
            super(view);
            tvOcrResult = view.findViewById(R.id.tvOcrResult);
            progressBar = view.findViewById(R.id.pbSingleItem);
            ivPreview = view.findViewById(R.id.ivPreview);
            btnSpellCheck = view.findViewById(R.id.btnSpellCheck);
        }
    }

    // Thêm hàm tính khoảng cách Levenshtein
    private int levenshtein(String a, String b) {
        int[][] dp = new int[a.length() + 1][b.length() + 1];
        for (int i = 0; i <= a.length(); i++) dp[i][0] = i;
        for (int j = 0; j <= b.length(); j++) dp[0][j] = j;
        for (int i = 1; i <= a.length(); i++) {
            for (int j = 1; j <= b.length(); j++) {
                if (a.charAt(i - 1) == b.charAt(j - 1))
                    dp[i][j] = dp[i - 1][j - 1];
                else
                    dp[i][j] = 1 + Math.min(dp[i - 1][j - 1], Math.min(dp[i - 1][j], dp[i][j - 1]));
            }
        }
        return dp[a.length()][b.length()];
    }

    private void showSpellCheckDialogWithTypos(String originalText, String fixedText, JSONArray typosArray, OcrResult result, int position, Button btnSpellCheck) {
        Log.d("SpellCheckDebug", "showSpellCheckDialogWithTypos START");
        AlertDialog.Builder builder = new AlertDialog.Builder(context);
        String title = context.getString(R.string.spell_check_result);
        String btnConfirm = context.getString(R.string.confirm);
        String btnCancel = context.getString(R.string.cancel);
        String btnUndo = context.getString(R.string.undo);
        String toastApplied = context.getString(R.string.spell_check_applied);
        String toastUndo = context.getString(R.string.undo_successful);
        builder.setTitle(title);
        ScrollView scrollView = new ScrollView(context);
        TextView textView = new TextView(context);
        textView.setTextIsSelectable(true);

        // Lưu trạng thái trước đó để Undo
        final String[] prevText = {originalText};
        final JSONArray[] prevTypos = {null};
        try {
            prevTypos[0] = typosArray != null ? new JSONArray(typosArray.toString()) : new JSONArray();
        } catch (org.json.JSONException e) {
            e.printStackTrace();
            prevTypos[0] = new JSONArray();
        }

        // Split the original text into lines to preserve structure
        String[] originalLines = originalText.split("\n");
        StringBuilder displayText = new StringBuilder();
        int charIndex = 0;
        for (String line : originalLines) {
            displayText.append(line);
            charIndex += line.length();
            if (charIndex < originalText.length()) {
                displayText.append("\n");
                charIndex++;
            }
        }

        SpannableString spannable = new SpannableString(displayText.toString());
        if (typosArray != null) {
            for (int i = 0; i < typosArray.length(); i++) {
                try {
                    JSONObject typo = typosArray.getJSONObject(i);
                    String word = typo.getString("word");
                    int start = typo.getInt("start");
                    int end = typo.getInt("end");
                    JSONArray suggestions = typo.getJSONArray("suggestions");

                    if (start >= 0 && end <= displayText.length()) {
                        spannable.setSpan(new ForegroundColorSpan(android.graphics.Color.RED), start, end, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);
                        final String[] suggestionList = new String[suggestions.length()];
                        for (int k = 0; k < suggestions.length(); k++) {
                            suggestionList[k] = suggestions.getString(k);
                        }
                        final String finalFixedText = displayText.toString();
                        final int finalStart = start;
                        final int finalEnd = end;
                        final OcrResult finalResult = result;
                        final int finalPosition = position;
                        spannable.setSpan(new ClickableSpan() {
                            @Override
                            public void onClick(View widget) {
                                // Lưu trạng thái trước khi sửa để Undo
                                prevText[0] = textView.getText().toString();
                                try {
                                    prevTypos[0] = new JSONArray(typosArray.toString());
                                } catch (org.json.JSONException e) {
                                    e.printStackTrace();
                                    prevTypos[0] = new JSONArray();
                                }
                                showSuggestionsDialog(word, suggestionList, finalFixedText, finalStart, finalEnd, finalResult, finalPosition, btnSpellCheck, prevText, prevTypos);
                            }
                        }, start, end, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);
                    }
                } catch (Exception e) {
                    Log.e("SpellCheckDebug", "Error processing typo: " + e.getMessage());
                }
            }
        }

        textView.setText(spannable);
        textView.setMovementMethod(LinkMovementMethod.getInstance());
        textView.setOnTouchListener((v, event) -> {
            if (event.getAction() == MotionEvent.ACTION_UP) {
                textView.setFocusable(false);
            }
            return false;
        });
        textView.setMinLines(6);
        textView.setMaxLines(20);
        scrollView.addView(textView);
        builder.setView(scrollView);
        builder.setPositiveButton(btnConfirm, (dialog, which) -> {
            Page oldPage = result.getPage();
            if (oldPage == null) {
                oldPage = new Page(
                    new java.util.ArrayList<>(),
                    position,
                    0,
                    0,
                    null,
                    fixedText
                );
            }
            Page newPage = new Page(
                oldPage.getBlocks(),
                oldPage.getPageIndex(),
                oldPage.getWidth(),
                oldPage.getHeight(),
                oldPage.getPreviewImage(),
                fixedText
            );
            result.setPage(newPage);
            notifyItemChanged(position);
            Toast.makeText(context, toastApplied, Toast.LENGTH_SHORT).show();
            btnSpellCheck.setEnabled(true);
        });
        builder.setNegativeButton(btnCancel, (dialog, which) -> {
            btnSpellCheck.setEnabled(true);
        });
        builder.setNeutralButton(btnUndo, (dialog, which) -> {
            showSpellCheckDialogWithTypos(prevText[0], prevText[0], prevTypos[0], result, position, btnSpellCheck);
            Toast.makeText(context, toastUndo, Toast.LENGTH_SHORT).show();
        });
        builder.setOnCancelListener(dialog -> btnSpellCheck.setEnabled(true));
        builder.show();
    }

    private void showSuggestionsDialog(String originalWord, String[] suggestions, String currentText, int start, int end, OcrResult result, int position, Button btnSpellCheck, String[] prevText, JSONArray[] prevTypos) {
        String chooseReplacement = context.getString(R.string.replacement_chosen, originalWord);
        AlertDialog.Builder builder = new AlertDialog.Builder(context);
        builder.setTitle(chooseReplacement);
        if (suggestions.length > 0) {
            builder.setItems(suggestions, (dialog, which) -> {
                String selectedSuggestion = suggestions[which];
                StringBuilder newText = new StringBuilder(currentText);
                newText.replace(start, end, selectedSuggestion);
                Page oldPage = result.getPage();
                if (oldPage == null) {
                    oldPage = new Page(
                        new java.util.ArrayList<>(),
                        position,
                        0,
                        0,
                        null,
                        newText.toString()
                    );
                }
                Page newPage = new Page(
                    oldPage.getBlocks(),
                    oldPage.getPageIndex(),
                    oldPage.getWidth(),
                    oldPage.getHeight(),
                    oldPage.getPreviewImage(),
                    newText.toString()
                );
                result.setPage(newPage);
                notifyItemChanged(position);
                Toast.makeText(context, "Replaced '" + originalWord + "' with '" + selectedSuggestion + "'", Toast.LENGTH_SHORT).show();
                recheckSpelling(newText.toString(), result, position, btnSpellCheck, isEnglishMode);
            });
        } else {
            builder.setMessage(context.getString(R.string.no_suggestions_available));
            builder.setPositiveButton(context.getString(R.string.ok), null);
        }
        builder.show();
    }

    // Thêm phương thức để gọi lại API kiểm tra chính tả
    private void recheckSpelling(String text, OcrResult result, int position, Button btnSpellCheck, boolean isEnglishMode) {
        JSONObject json = new JSONObject();
        try {
            json.put("text", text);
            json.put("language", isEnglishMode ? "en_US" : "vi_VN");
        } catch (Exception e) {
            Toast.makeText(context, "Lỗi tạo request: " + e.getMessage(), Toast.LENGTH_LONG).show();
            btnSpellCheck.setEnabled(true);
            return;
        }
        String spellcheckUrl = MainActivity.BASE_URL + "/spellcheck";
        RequestBody body = RequestBody.create(json.toString(), MediaType.get("application/json; charset=utf-8"));
        Request request = new Request.Builder()
                .url(spellcheckUrl)
                .post(body)
                .build();
        client.newCall(request).enqueue(new Callback() {
            @Override
            public void onFailure(Call call, IOException e) {
                Log.e("SpellCheckDebug", "onFailure: " + e.getMessage());
                ((android.app.Activity) context).runOnUiThread(() -> {
                    Toast.makeText(context, "Cần internet để sửa lỗi chính tả", Toast.LENGTH_LONG).show();
                    btnSpellCheck.setEnabled(true);
                });
            }
            @Override
            public void onResponse(Call call, Response response) throws IOException {
                try {
                    Log.d("SpellCheckDebug", "onResponse START. Successful: " + response.isSuccessful());

                    if (!response.isSuccessful()) {
                        Log.e("SpellCheckDebug", "Response not successful. Code: " + response.code());
                        ((android.app.Activity) context).runOnUiThread(() -> {
                            Toast.makeText(context, "Lỗi server: " + response.message(), Toast.LENGTH_LONG).show();
                            btnSpellCheck.setEnabled(true);
                        });
                        return;
                    }

                    final String responseBody = response.body().string();
                    Log.d("SpellCheckDebug", "Successfully read response body. Length: " + responseBody.length());

                    final JSONObject json = new JSONObject(responseBody);
                    final String fixedText = json.optString("corrected_text", text);
                    final JSONArray typosArray = json.optJSONArray("typos");
                    Log.d("SpellCheckDebug", "Parsed JSON, got fixedText. Length: " + fixedText.length());

                    ((android.app.Activity) context).runOnUiThread(() -> {
                        try {
                            Log.d("SpellCheckDebug", "Inside runOnUiThread. Preparing to show dialog.");
                            Toast.makeText(context, "Đã nhận kết quả, đang hiển thị...", Toast.LENGTH_SHORT).show();
                            showSpellCheckDialogWithTypos(text, fixedText, typosArray, result, position, btnSpellCheck);
                            Log.d("SpellCheckDebug", "showSpellCheckDialog call completed.");
                        } catch (Exception e) {
                            Log.e("SpellCheckDebug", "ERROR inside runOnUiThread (showing dialog)", e);
                            Toast.makeText(context, "Lỗi hiển thị kết quả: " + e.getMessage(), Toast.LENGTH_LONG).show();
                            btnSpellCheck.setEnabled(true);
                        }
                    });

                } catch (Exception e) {
                    Log.e("SpellCheckDebug", "FATAL ERROR in onResponse background task", e);
                    ((android.app.Activity) context).runOnUiThread(() -> {
                        Toast.makeText(context, "Lỗi xử lý phản hồi: " + e.getMessage(), Toast.LENGTH_LONG).show();
                        btnSpellCheck.setEnabled(true);
                    });
                }
            }
        });
    }
} 