package com.example.myocr;

import android.content.Context;
import android.graphics.Bitmap;
import android.net.Uri;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;
import com.bumptech.glide.Glide;
import java.util.List;

public class ThumbnailAdapter extends RecyclerView.Adapter<ThumbnailAdapter.ThumbnailViewHolder> {

    private final Context context;
    private final List<Bitmap> thumbnails;
    private final OnThumbnailClickListener listener;
    private int selectedPosition = 0;

    public interface OnThumbnailClickListener {
        void onThumbnailClick(int position);
    }

    public ThumbnailAdapter(Context context, List<Bitmap> thumbnails, OnThumbnailClickListener listener) {
        this.context = context;
        this.thumbnails = thumbnails;
        this.listener = listener;
    }

    @NonNull
    @Override
    public ThumbnailViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(context).inflate(R.layout.item_thumbnail, parent, false);
        return new ThumbnailViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull ThumbnailViewHolder holder, int position) {
        Bitmap thumbnail = thumbnails.get(position);
        Glide.with(context).load(thumbnail).into(holder.imageView);
        holder.selectionOverlay.setVisibility(position == selectedPosition ? View.VISIBLE : View.GONE);
    }

    @Override
    public int getItemCount() {
        return thumbnails.size();
    }

    public void setSelectedPosition(int position) {
        if (position < 0 || position >= thumbnails.size()) return;
        int oldPosition = selectedPosition;
        selectedPosition = position;
        notifyItemChanged(oldPosition);
        notifyItemChanged(selectedPosition);
    }
    
    public int getSelectedPosition() {
        return selectedPosition;
    }

    class ThumbnailViewHolder extends RecyclerView.ViewHolder {
        ImageView imageView;
        View selectionOverlay;

        ThumbnailViewHolder(View itemView) {
            super(itemView);
            imageView = itemView.findViewById(R.id.thumbnail_image_view);
            selectionOverlay = itemView.findViewById(R.id.thumbnail_selection_overlay);
            itemView.setOnClickListener(v -> {
                if (listener != null) {
                    int position = getAdapterPosition();
                    if (position != RecyclerView.NO_POSITION) {
                        listener.onThumbnailClick(position);
                    }
                }
            });
        }
    }
} 