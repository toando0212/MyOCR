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

public class OcrResultAdapter extends RecyclerView.Adapter<OcrResultAdapter.ViewHolder> {

    private final Context context;
    private List<OcrResult> ocrResultList;
    private boolean isEnglishMode = false;
    private static HashSet<String> vietnameseDict;
    private static boolean dictLoaded = false;

    public OcrResultAdapter(Context context, List<OcrResult> ocrResultList, boolean isEnglishMode) {
        this.context = context;
        this.ocrResultList = ocrResultList;
        this.isEnglishMode = isEnglishMode;
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
                    OkHttpClient client = new OkHttpClient();
                    String originalText = result.getText();
                    RequestBody formBody = new FormBody.Builder()
                            .add("language", "en-US")
                            .add("text", originalText)
                            .build();
                    Request request = new Request.Builder()
                            .url("https://api.languagetool.org/v2/check")
                            .post(formBody)
                            .build();
                    client.newCall(request).enqueue(new Callback() {
                        @Override
                        public void onFailure(Call call, IOException e) {
                            ((android.app.Activity) context).runOnUiThread(() -> {
                                String msg = Locale.getDefault().getLanguage().equals("vi") ? "Cần internet để sửa lỗi chính tả" : "Internet is required for spell check";
                                Toast.makeText(context, msg, Toast.LENGTH_LONG).show();
                                btnSpellCheck.setEnabled(true);
                            });
                        }
                        @Override
                        public void onResponse(Call call, Response response) throws IOException {
                            if (!response.isSuccessful()) {
                                ((android.app.Activity) context).runOnUiThread(() -> {
                                    Toast.makeText(context, "Lỗi khi sửa chính tả: " + response.message(), Toast.LENGTH_LONG).show();
                                    btnSpellCheck.setEnabled(true);
                                });
                                return;
                            }
                            String responseBody = response.body().string();
                            try {
                                JSONObject json = new JSONObject(responseBody);
                                JSONArray matches = json.getJSONArray("matches");
                                StringBuilder corrected = new StringBuilder(originalText);
                                int offset = 0;
                                for (int i = 0; i < matches.length(); i++) {
                                    JSONObject match = matches.getJSONObject(i);
                                    JSONArray replacements = match.getJSONArray("replacements");
                                    if (replacements.length() > 0) {
                                        String replacement = replacements.getJSONObject(0).getString("value");
                                        int fromPos = match.getInt("offset") + offset;
                                        int toPos = fromPos + match.getInt("length");
                                        corrected.replace(fromPos, toPos, replacement);
                                        offset += replacement.length() - match.getInt("length");
                                    }
                                }
                                String fixedText = corrected.toString();
                                // Hiện dialog xác nhận
                                ((android.app.Activity) context).runOnUiThread(() -> {
                                    showSpellCheckDialog(originalText, fixedText, result, position, btnSpellCheck);
                                });
                            } catch (Exception e) {
                                ((android.app.Activity) context).runOnUiThread(() -> {
                                    Toast.makeText(context, "Lỗi khi phân tích kết quả sửa chính tả: " + e.getMessage(), Toast.LENGTH_LONG).show();
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
                    // Load dictionary nếu chưa load
                    if (!dictLoaded) {
                        vietnameseDict = new HashSet<>();
                        try {
                            AssetManager am = context.getAssets();
                            InputStream is = am.open("Viet74K.txt");
                            BufferedReader reader = new BufferedReader(new InputStreamReader(is, "UTF-8"));
                            String line;
                            while ((line = reader.readLine()) != null) {
                                vietnameseDict.add(line.trim().toLowerCase());
                            }
                            reader.close();
                            is.close();
                            dictLoaded = true;
                        } catch (Exception e) {
                            ((android.app.Activity) context).runOnUiThread(() -> {
                                Toast.makeText(context, "Lỗi tải từ điển: " + e.getMessage(), Toast.LENGTH_LONG).show();
                                btnSpellCheck.setEnabled(true);
                            });
                            return;
                        }
                    }
                    // Giữ nguyên line breaks khi sửa
                    String originalText = result.getText();
                    String[] lines = originalText.split("\\r?\\n");
                    StringBuilder checked = new StringBuilder();
                    for (int l = 0; l < lines.length; l++) {
                        String line = lines[l];
                        String[] words = line.split("\\s+");
                        for (String word : words) {
                            String cleanWord = word.replaceAll("[^\\p{L}]", "").toLowerCase();
                            if (cleanWord.isEmpty()) {
                                checked.append(word).append(" ");
                                continue;
                            }
                            if (vietnameseDict.contains(cleanWord)) {
                                checked.append(word).append(" ");
                            } else {
                                // Tìm từ gần đúng nhất trong từ điển
                                String best = null;
                                int minDist = Integer.MAX_VALUE;
                                for (String dictWord : vietnameseDict) {
                                    int dist = levenshtein(cleanWord, dictWord);
                                    if (dist < minDist) {
                                        minDist = dist;
                                        best = dictWord;
                                        if (minDist == 1) break;
                                    }
                                }
                                if (best != null && minDist <= 2) {
                                    String suggest = best;
                                    if (Character.isUpperCase(word.charAt(0))) {
                                        suggest = Character.toUpperCase(best.charAt(0)) + best.substring(1);
                                    }
                                    checked.append(suggest).append(" ");
                                } else {
                                    checked.append(word).append(" ");
                                }
                            }
                        }
                        if (l < lines.length - 1) checked.append("\n");
                    }
                    String fixedText = checked.toString().trim();
                    // Hiện dialog xác nhận
                    ((android.app.Activity) context).runOnUiThread(() -> {
                        showSpellCheckDialog(originalText, fixedText, result, position, btnSpellCheck);
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

    // Thêm hàm showSpellCheckDialog
    private void showSpellCheckDialog(String originalText, String fixedText, OcrResult result, int position, Button btnSpellCheck) {
        AlertDialog.Builder builder = new AlertDialog.Builder(context);
        builder.setTitle("Kết quả sửa lỗi chính tả");
        ScrollView scrollView = new ScrollView(context);
        EditText editText = new EditText(context);
        editText.setText(fixedText);
        editText.setMinLines(6);
        editText.setMaxLines(20);
        editText.setTextIsSelectable(true);
        editText.setFocusable(false);
        scrollView.addView(editText);
        builder.setView(scrollView);
        builder.setPositiveButton("Xác nhận", (dialog, which) -> {
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
            Toast.makeText(context, "Đã áp dụng sửa lỗi chính tả!", Toast.LENGTH_SHORT).show();
            btnSpellCheck.setEnabled(true);
        });
        builder.setNegativeButton("Hủy", (dialog, which) -> {
            // Không thay đổi gì
            btnSpellCheck.setEnabled(true);
        });
        builder.setOnCancelListener(dialog -> btnSpellCheck.setEnabled(true));
        builder.show();
    }
} 