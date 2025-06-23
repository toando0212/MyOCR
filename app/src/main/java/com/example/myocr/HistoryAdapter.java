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
        private final ImageView previewImageView;
        private final TextView timestampTextView;
        private final TextView imageCountTextView;
        private final TextView fullTextView;
        private final ImageView deleteButton;

        ViewHolder(View itemView) {
            super(itemView);
            previewImageView = itemView.findViewById(R.id.history_preview_image);
            timestampTextView = itemView.findViewById(R.id.history_timestamp);
            imageCountTextView = itemView.findViewById(R.id.history_image_count);
            fullTextView = itemView.findViewById(R.id.history_full_text);
            deleteButton = itemView.findViewById(R.id.history_delete_button);
        }

        void bind(final HistorySession session, final OnHistorySessionInteractionListener listener, final int position) {
            timestampTextView.setText(session.getTimestamp());
            imageCountTextView.setText(context.getResources().getString(R.string.image_count, session.getImageCount()));
            fullTextView.setText(session.getFullTextPreview());

            Glide.with(context)
                    .load(session.getPreviewImageUri())
                    .placeholder(R.drawable.ic_image_placeholder)
                    .error(R.drawable.ic_image_placeholder)
                    .into(previewImageView);

            itemView.setOnClickListener(v -> listener.onSessionClick(session));
            deleteButton.setOnClickListener(v -> listener.onDeleteSessionClick(session, position));
        }
    }
} 