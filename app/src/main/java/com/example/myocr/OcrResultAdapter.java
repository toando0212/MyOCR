package com.example.myocr;

import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.core.content.ContextCompat;
import androidx.recyclerview.widget.RecyclerView;

import com.bumptech.glide.Glide;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.List;

public class OcrResultAdapter extends RecyclerView.Adapter<OcrResultAdapter.ViewHolder> {

    private final Context context;
    private List<OcrResult> ocrResultList;

    public OcrResultAdapter(Context context, List<OcrResult> ocrResultList) {
        this.context = context;
        this.ocrResultList = ocrResultList;
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
    }

    @Override
    public int getItemCount() {
        return ocrResultList.size();
    }

    public static class ViewHolder extends RecyclerView.ViewHolder {
        TextView tvOcrResult;
        ProgressBar progressBar;
        ImageView ivPreview;

        public ViewHolder(View view) {
            super(view);
            tvOcrResult = view.findViewById(R.id.tvOcrResult);
            progressBar = view.findViewById(R.id.pbSingleItem);
            ivPreview = view.findViewById(R.id.ivPreview);
        }
    }
} 