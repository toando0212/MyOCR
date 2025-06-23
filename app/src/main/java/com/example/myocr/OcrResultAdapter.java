package com.example.myocr;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import com.bumptech.glide.Glide;

import java.util.List;

public class OcrResultAdapter extends RecyclerView.Adapter<OcrResultAdapter.ViewHolder> {

    private final Context context;
    private List<OcrResult> ocrResults;

    public OcrResultAdapter(Context context, List<OcrResult> ocrResults) {
        this.context = context;
        this.ocrResults = ocrResults;
    }

    public void setOcrResults(List<OcrResult> ocrResults) {
        this.ocrResults = ocrResults;
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
        OcrResult result = ocrResults.get(position);
        holder.bind(result);
    }

    @Override
    public int getItemCount() {
        return ocrResults.size();
    }

    class ViewHolder extends RecyclerView.ViewHolder {
        private final ImageView resultImageView;
        private final TextView resultTextView;

        ViewHolder(View itemView) {
            super(itemView);
            resultImageView = itemView.findViewById(R.id.resultImageView);
            resultTextView = itemView.findViewById(R.id.resultTextView);
        }

        void bind(OcrResult result) {
            Glide.with(context)
                    .load(result.getImageUri())
                    .placeholder(R.drawable.ic_image_placeholder) // Optional placeholder
                    .error(R.drawable.ic_image_placeholder)       // Optional error image
                    .into(resultImageView);

            if (result.isProcessing()) {
                resultTextView.setText(R.string.processing);
            } else {
                resultTextView.setText(result.getRecognizedText());
            }
        }
    }
} 