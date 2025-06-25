package com.example.myocr;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.PointF;
import android.graphics.Bitmap;
import android.util.AttributeSet;
import android.view.MotionEvent;
import android.view.View;

public class PolygonView extends View {
    private PointF[] points = new PointF[4];
    private int selectedPoint = -1;
    private float radius = 30f;
    private Paint pointPaint, linePaint;

    public PolygonView(Context context, AttributeSet attrs) {
        super(context, attrs);
        pointPaint = new Paint();
        pointPaint.setColor(Color.RED);
        pointPaint.setStyle(Paint.Style.FILL);

        linePaint = new Paint();
        linePaint.setColor(Color.GREEN);
        linePaint.setStrokeWidth(5f);
    }

    public void setDefaultCorners(int viewWidth, int viewHeight) {
        // Fallback if auto-detection fails
        points[0] = new PointF(viewWidth * 0.2f, viewHeight * 0.2f);
        points[1] = new PointF(viewWidth * 0.8f, viewHeight * 0.2f);
        points[2] = new PointF(viewWidth * 0.8f, viewHeight * 0.8f);
        points[3] = new PointF(viewWidth * 0.2f, viewHeight * 0.8f);
        invalidate();
    }

    public void setPoints(PointF[] pts) {
        if (pts != null && pts.length == 4) {
            for (int i = 0; i < 4; i++) points[i] = pts[i];
            invalidate();
        }
    }

    public PointF[] getPoints() { return points; }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        if (points[0] == null) return; // Don't draw if points are not initialized
        // Vẽ các cạnh
        for (int i = 0; i < 4; i++) {
            PointF p1 = points[i];
            PointF p2 = points[(i+1)%4];
            canvas.drawLine(p1.x, p1.y, p2.x, p2.y, linePaint);
        }
        // Vẽ các điểm
        for (PointF p : points) {
            canvas.drawCircle(p.x, p.y, radius, pointPaint);
        }
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        float x = event.getX(), y = event.getY();
        switch (event.getAction()) {
            case MotionEvent.ACTION_DOWN:
                for (int i = 0; i < 4; i++) {
                    if (Math.hypot(points[i].x - x, points[i].y - y) < radius * 2) {
                        selectedPoint = i;
                        break;
                    }
                }
                break;
            case MotionEvent.ACTION_MOVE:
                if (selectedPoint != -1) {
                    points[selectedPoint].x = x;
                    points[selectedPoint].y = y;
                    invalidate();
                }
                break;
            case MotionEvent.ACTION_UP:
                selectedPoint = -1;
                break;
        }
        return true;
    }
}
