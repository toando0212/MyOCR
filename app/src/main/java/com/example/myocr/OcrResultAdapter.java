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
import android.text.SpannableStringBuilder;
import android.text.Spanned;
import android.text.style.ForegroundColorSpan;
import android.text.style.ClickableSpan;
import android.view.MotionEvent;
import android.text.method.LinkMovementMethod;
import java.util.ArrayList;

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
            } else if (result.getText() != null && !result.getText().isEmpty()) {
                // This is the new case: For results loaded from history that have text but no Page object.
                holder.tvOcrResult.setText(result.getText());
                holder.tvOcrResult.setTextColor(ContextCompat.getColor(context, android.R.color.black));
                // No preview image is available in this case
                holder.ivPreview.setVisibility(View.GONE);
            } else {
                // Handle case where processing is done but page is null and no error
                holder.tvOcrResult.setText(context.getString(R.string.no_results_available));
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
        AlertDialog.Builder builder = new AlertDialog.Builder(context);
        String title = context.getString(R.string.spell_check_results_title);
        String toastApplied = context.getString(R.string.spell_check_applied);
        String toastUndo = context.getString(R.string.undo_successful);
        builder.setTitle(title);

        ScrollView scrollView = new ScrollView(context);
        TextView textView = new TextView(context);
        textView.setTextIsSelectable(true);
        textView.setPadding(48, 16, 48, 16); // Add some padding for better look
        textView.setTextSize(16f);
        
        // This will hold the text content and all the clickable spans
        SpannableStringBuilder spannableBuilder = new SpannableStringBuilder(originalText);
        
        // We will pass this 'typosArray' to the suggestion dialog so it can be modified
        final List<JSONObject> typoList = new ArrayList<>();
        if (typosArray != null) {
            for (int i = 0; i < typosArray.length(); i++) {
                try {
                    typoList.add(typosArray.getJSONObject(i));
                } catch (Exception e) {}
            }
        }
        
        // Apply spans for all typos
        applyTyposToSpannable(spannableBuilder, typoList, textView, result, position, btnSpellCheck);
        
        textView.setText(spannableBuilder);
        textView.setMovementMethod(LinkMovementMethod.getInstance());
        scrollView.addView(textView);
        builder.setView(scrollView);

        // Apply Button
        builder.setPositiveButton(R.string.apply_changes, (dialog, which) -> {
            String correctedText = textView.getText().toString();
            if (result.getPage() != null) {
                result.getPage().updateContent(correctedText);
            }
            notifyItemChanged(position);
            Toast.makeText(context, toastApplied, Toast.LENGTH_SHORT).show();
            btnSpellCheck.setEnabled(true);
        });

        // Undo Button
        builder.setNeutralButton(R.string.undo_changes, (dialog, which) -> {
            // This button's action is set to null here. We override it later to prevent the dialog from closing.
        });

        // Cancel Button
        builder.setNegativeButton(android.R.string.cancel, (dialog, which) -> {
            btnSpellCheck.setEnabled(true); // Re-enable the button on cancel
            dialog.dismiss();
        });

        final AlertDialog dialog = builder.create();

        dialog.setOnShowListener(d -> {
            Button neutralButton = dialog.getButton(AlertDialog.BUTTON_NEUTRAL);
            neutralButton.setOnClickListener(v -> {
                // Manually reset the text and re-apply the spans without dismissing the dialog
                spannableBuilder.clear();
                spannableBuilder.append(originalText);
                typoList.clear(); // Clear the list of applied corrections
                try {
                    for (int i = 0; i < typosArray.length(); i++) {
                        typoList.add(typosArray.getJSONObject(i));
                    }
                } catch (Exception e) {
                    // This will be caught by the outer try-catch
                }
                applyTyposToSpannable(spannableBuilder, typoList, textView, result, position, btnSpellCheck);
                textView.setText(spannableBuilder);
                Toast.makeText(context, toastUndo, Toast.LENGTH_SHORT).show();
            });
        });


        dialog.show();
    }
    
    // Helper method to apply all typo spans to the text
    private void applyTyposToSpannable(SpannableStringBuilder spannableBuilder, List<JSONObject> typoList, TextView textView, OcrResult result, int position, Button btnSpellCheck) {
        // First, clear any old spans
        ClickableSpan[] oldClickableSpans = spannableBuilder.getSpans(0, spannableBuilder.length(), ClickableSpan.class);
        for (ClickableSpan span : oldClickableSpans) {
            spannableBuilder.removeSpan(span);
        }
        ForegroundColorSpan[] oldColorSpans = spannableBuilder.getSpans(0, spannableBuilder.length(), ForegroundColorSpan.class);
        for (ForegroundColorSpan span : oldColorSpans) {
            spannableBuilder.removeSpan(span);
        }
        
        for (int i = 0; i < typoList.size(); i++) {
            final int typoIndex = i;
            try {
                JSONObject typo = typoList.get(i);
                final String word = typo.getString("word");
                int start = typo.getInt("start");
                int end = typo.getInt("end");
                JSONArray suggestions = typo.getJSONArray("suggestions");

                if (start >= 0 && end <= spannableBuilder.length()) {
                    spannableBuilder.setSpan(new ForegroundColorSpan(android.graphics.Color.RED), start, end, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);
                    
                    ClickableSpan clickableSpan = new ClickableSpan() {
                        @Override
                        public void onClick(View widget) {
                            showSuggestionsDialog(word, suggestions, textView, spannableBuilder, typoList, typoIndex, result, position, btnSpellCheck);
                        }
                    };
                    spannableBuilder.setSpan(clickableSpan, start, end, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);
                }
            } catch (Exception e) {
                Log.e("SpellCheckDebug", "Error applying typo span: " + e.getMessage());
            }
        }
    }

    private void showSuggestionsDialog(String originalWord, JSONArray suggestions, TextView mainTextView, SpannableStringBuilder spannableBuilder, List<JSONObject> typoList, int typoIndex, OcrResult result, int position, Button btnSpellCheck) {
        String chooseReplacement = context.getString(R.string.replacement_chosen, originalWord);
        AlertDialog.Builder builder = new AlertDialog.Builder(context);
        builder.setTitle(chooseReplacement);
        
        try {
            final String[] suggestionItems = new String[suggestions.length()];
            for (int k = 0; k < suggestions.length(); k++) {
                suggestionItems[k] = suggestions.getString(k);
            }
            
            if (suggestionItems.length > 0) {
                builder.setItems(suggestionItems, (dialog, which) -> {
                    String selectedSuggestion = suggestionItems[which];
                    try {
                        // Get the typo object to find its position
                        JSONObject typoToFix = typoList.get(typoIndex);
                        int start = typoToFix.getInt("start");
                        int end = typoToFix.getInt("end");
                        // Replace the word in the SpannableStringBuilder
                        spannableBuilder.replace(start, end, selectedSuggestion);
                        // Remove the fixed typo from our list
                        typoList.remove(typoIndex);
                        // Update the indices of all subsequent typos
                        int delta = selectedSuggestion.length() - originalWord.length();
                        for (int i = typoIndex; i < typoList.size(); i++) { // Start from the current index
                            JSONObject subsequentTypo = typoList.get(i);
                            try {
                                subsequentTypo.put("start", subsequentTypo.getInt("start") + delta);
                                subsequentTypo.put("end", subsequentTypo.getInt("end") + delta);
                            } catch (org.json.JSONException e) {
                                Log.e("SpellCheckDebug", "Error updating typo indices: " + e.getMessage());
                            }
                        }
                        // Re-apply all spans to the updated builder
                        applyTyposToSpannable(spannableBuilder, typoList, mainTextView, result, position, btnSpellCheck);
                        // Update the TextView in the main dialog
                        mainTextView.setText(spannableBuilder);
                        Toast.makeText(context, "Replaced '" + originalWord + "' with '" + selectedSuggestion + "'", Toast.LENGTH_SHORT).show();
                    } catch (org.json.JSONException e) {
                        Log.e("SpellCheckDebug", "Error handling suggestion click: " + e.getMessage());
                        Toast.makeText(context, "Error applying suggestion.", Toast.LENGTH_SHORT).show();
                    }
                });
            } else {
                builder.setMessage(context.getString(R.string.no_suggestions_available));
                builder.setPositiveButton(context.getString(R.string.ok), null);
            }
        } catch (Exception e) {
             Log.e("SpellCheckDebug", "Error showing suggestions: " + e.getMessage());
             builder.setMessage("Error loading suggestions.");
             builder.setPositiveButton(context.getString(R.string.ok), null);
        }
        
        builder.show();
    }
} 