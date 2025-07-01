package com.example.myocr;

import android.content.Context;
import android.net.Uri;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.annotation.NonNull;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.bumptech.glide.Glide;
import com.bumptech.glide.load.engine.DiskCacheStrategy;
import java.util.List;

public class HistoryAdapter extends RecyclerView.Adapter<HistoryAdapter.ViewHolder> {

    private final Context context;
    private List<HistorySession> sessions;
    private OnHistorySessionInteractionListener listener;

    public interface OnHistorySessionInteractionListener {
        void onSessionClick(HistorySession session);
        void onDeleteSessionClick(HistorySession session, int position);
    }

    public HistoryAdapter(Context context, List<HistorySession> sessions, OnHistorySessionInteractionListener listener) {
        this.context = context;
        this.sessions = sessions;
        this.listener = listener;
    }

    public void setSessions(List<HistorySession> sessions) {
        this.sessions = sessions;
        notifyDataSetChanged();
    }

    @NonNull
    @Override
    public ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(context).inflate(R.layout.item_history_session, parent, false);
        return new ViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull ViewHolder holder, int position) {
        HistorySession session = sessions.get(position);
        holder.bind(session, listener, position);
    }

    @Override
    public int getItemCount() {
        return sessions.size();
    }

    class ViewHolder extends RecyclerView.ViewHolder {
        private final RecyclerView previewRecyclerView;
        private final TextView timestampTextView;
        private final TextView imageCountTextView;
        private final TextView fullTextView;
        private final ImageView deleteButton;

        ViewHolder(View itemView) {
            super(itemView);
            previewRecyclerView = itemView.findViewById(R.id.history_preview_recycler);
            timestampTextView = itemView.findViewById(R.id.history_timestamp);
            imageCountTextView = itemView.findViewById(R.id.history_image_count);
            fullTextView = itemView.findViewById(R.id.history_full_text);
            deleteButton = itemView.findViewById(R.id.history_delete_button);
        }

        void bind(final HistorySession session, final OnHistorySessionInteractionListener listener, final int position) {
            timestampTextView.setText(session.getTimestamp());
            imageCountTextView.setText(context.getResources().getString(R.string.image_count, session.getImageCount()));
            fullTextView.setText(session.getFullTextPreview());

            // Set up the horizontal RecyclerView for image previews
            List<HistoryItemDetail> details = session.getDetails();
            List<Uri> imageUris = new java.util.ArrayList<>();
            if (details != null) {
                for (HistoryItemDetail detail : details) {
                    imageUris.add(detail.getImageUri());
                }
            }
            HistoryPreviewAdapter previewAdapter = new HistoryPreviewAdapter(context, imageUris);
            previewRecyclerView.setLayoutManager(new LinearLayoutManager(context, LinearLayoutManager.HORIZONTAL, false));
            previewRecyclerView.setAdapter(previewAdapter);

            itemView.setOnClickListener(v -> listener.onSessionClick(session));
            deleteButton.setOnClickListener(v -> listener.onDeleteSessionClick(session, position));
        }
    }

    // Adapter for displaying image thumbnails in the history preview
    private static class HistoryPreviewAdapter extends RecyclerView.Adapter<HistoryPreviewAdapter.PreviewViewHolder> {
        private final Context context;
        private final List<Uri> imageUris;

        HistoryPreviewAdapter(Context context, List<Uri> imageUris) {
            this.context = context;
            this.imageUris = imageUris;
        }

        @NonNull
        @Override
        public PreviewViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
            View view = LayoutInflater.from(context).inflate(R.layout.item_thumbnail, parent, false);
            return new PreviewViewHolder(view);
        }

        @Override
        public void onBindViewHolder(@NonNull PreviewViewHolder holder, int position) {
            Uri uri = imageUris.get(position);
            Glide.with(context)
                    .load(uri)
                    .diskCacheStrategy(DiskCacheStrategy.NONE)
                    .skipMemoryCache(true)
                    .into(holder.imageView);
            holder.selectionOverlay.setVisibility(View.GONE); // No selection in history preview
        }

        @Override
        public int getItemCount() {
            return imageUris.size();
        }

        static class PreviewViewHolder extends RecyclerView.ViewHolder {
            ImageView imageView;
            View selectionOverlay;
            PreviewViewHolder(View itemView) {
                super(itemView);
                imageView = itemView.findViewById(R.id.thumbnail_image_view);
                selectionOverlay = itemView.findViewById(R.id.thumbnail_selection_overlay);
            }
        }
    }
} 