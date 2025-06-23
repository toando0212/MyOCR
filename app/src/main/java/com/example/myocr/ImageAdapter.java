package com.example.myocr;

import android.content.Context;
import android.net.Uri;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.ImageButton;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.recyclerview.widget.RecyclerView;

import java.util.List;

public class ImageAdapter extends RecyclerView.Adapter<ImageAdapter.ViewHolder> {
    private List<Uri> imageUris;
    private Context context;
    private OnImageClickListener listener;

    public interface OnImageClickListener {
        void onImageClick(Uri imageUri);
        void onImageLongClick(Uri imageUri);
    }

    public ImageAdapter(Context context, List<Uri> imageUris, OnImageClickListener listener) {
        this.context = context;
        this.imageUris = imageUris;
        this.listener = listener;
    }

    @NonNull
    @Override
    public ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(context).inflate(R.layout.item_image, parent, false);
        return new ViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull ViewHolder holder, int position) {
        Uri uri = imageUris.get(position);
        holder.imageView.setImageURI(uri);

        holder.imageView.setOnClickListener(v -> {
            if (listener != null) {
                listener.onImageClick(uri);
            }
        });

        holder.imageView.setOnLongClickListener(v -> {
            if (listener != null) {
                listener.onImageLongClick(uri);
            }
            return true;
        });
    }

    @Override
    public int getItemCount() {
        return imageUris.size();
    }

    public static class ViewHolder extends RecyclerView.ViewHolder {
        ImageView imageView;

        public ViewHolder(@NonNull View itemView) {
            super(itemView);
            imageView = itemView.findViewById(R.id.imageView);
        }
    }

    public void setImageUris(List<Uri> imageUris) {
        this.imageUris = imageUris;
        notifyDataSetChanged();
    }
} 