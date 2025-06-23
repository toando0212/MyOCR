package com.example.myocr;

import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import com.google.android.material.snackbar.Snackbar;
import com.google.android.material.textfield.TextInputEditText;
import com.google.android.material.textfield.TextInputLayout;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.IOException;
import java.util.concurrent.TimeUnit;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

public class LoginActivity extends AppCompatActivity {

    private TextInputEditText etUsername, etPassword, etConfirmPassword;
    private TextInputLayout tilUsername, tilPassword, tilConfirmPassword;
    private Button btnAuth, btnGuest;
    private TextView tvToggleMode;
    private boolean isLoginMode = true;
    private final OkHttpClient client = new OkHttpClient.Builder()
            .connectTimeout(60, TimeUnit.SECONDS)
            .writeTimeout(60, TimeUnit.SECONDS)
            .readTimeout(60, TimeUnit.SECONDS)
            .build();
    private static final String BASE_URL = "https://7c2c-2405-4803-f801-12a0-1883-6ffe-89a4-5660.ngrok-free.app"; // IMPORTANT: Use your actual server URL

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_login);

        // Initialize views
        etUsername = findViewById(R.id.etUsername);
        etPassword = findViewById(R.id.etPassword);
        etConfirmPassword = findViewById(R.id.etConfirmPassword);
        tilUsername = findViewById(R.id.tilUsername);
        tilPassword = findViewById(R.id.tilPassword);
        tilConfirmPassword = findViewById(R.id.tilConfirmPassword);
        btnAuth = findViewById(R.id.btnAuth);
        btnGuest = findViewById(R.id.btnGuest);
        tvToggleMode = findViewById(R.id.tvToggleMode);

        updateUIForMode();

        tvToggleMode.setOnClickListener(v -> {
            isLoginMode = !isLoginMode;
            updateUIForMode();
        });

        btnAuth.setOnClickListener(v -> {
            String username = etUsername.getText().toString().trim();
            String password = etPassword.getText().toString().trim();
            String confirmPassword = etConfirmPassword.getText().toString().trim();

            // Basic validation
            if (username.isEmpty() || password.isEmpty()) {
                showSnackbar("Username and password cannot be empty.");
                return;
            }
            if (!isLoginMode && !password.equals(confirmPassword)) {
                showSnackbar("Passwords do not match.");
                return;
            }

            setLoading(true);
            if (isLoginMode) {
                authenticateUser("/login", username, password);
            } else {
                authenticateUser("/register", username, password);
            }
        });

        btnGuest.setOnClickListener(v -> {
            SharedPreferences prefs = getSharedPreferences("user_prefs", MODE_PRIVATE);
            prefs.edit().clear().apply(); // Clear all user data for guest mode
            navigateToMain();
        });
    }

    private void authenticateUser(String endpoint, String username, String password) {
        MediaType JSON = MediaType.get("application/json; charset=utf-8");
        JSONObject jsonObject = new JSONObject();
        try {
            jsonObject.put("username", username);
            jsonObject.put("password", password);
        } catch (JSONException e) {
            e.printStackTrace();
            setLoading(false);
            return;
        }

        RequestBody body = RequestBody.create(jsonObject.toString(), JSON);
        Request request = new Request.Builder()
                .url(BASE_URL + endpoint)
                .post(body)
                .build();

        client.newCall(request).enqueue(new Callback() {
            @Override
            public void onFailure(@NonNull Call call, @NonNull IOException e) {
                runOnUiThread(() -> {
                    setLoading(false);
                    showSnackbar("Network error: " + e.getMessage());
                });
            }

            @Override
            public void onResponse(@NonNull Call call, @NonNull Response response) throws IOException {
                final String responseBody = response.body().string();
                runOnUiThread(() -> {
                    setLoading(false);
                    try {
                        JSONObject json = new JSONObject(responseBody);
                        if (response.isSuccessful()) {
                            if (endpoint.equals("/login")) {
                                int userId = json.getInt("user_id");
                                saveUserSession(userId);
                                showSnackbar("Login successful!");
                                navigateToMain();
                            } else { // Registration
                                showSnackbar("Registration successful! Please log in.");
                                isLoginMode = true;
                                updateUIForMode();
                            }
                        } else {
                            String error = json.optString("error", "An unknown error occurred.");
                            showSnackbar(error);
                        }
                    } catch (JSONException e) {
                        showSnackbar("Failed to parse server response.");
                    }
                });
            }
        });
    }

    private void saveUserSession(int userId) {
        SharedPreferences prefs = getSharedPreferences("user_prefs", MODE_PRIVATE);
        prefs.edit()
                .putBoolean("isLoggedIn", true)
                .putInt("userId", userId)
                .apply();
    }

    private void updateUIForMode() {
        if (isLoginMode) {
            tilConfirmPassword.setVisibility(View.GONE);
            btnAuth.setText(R.string.login);
            tvToggleMode.setText("Don't have an account? Register");
        } else {
            tilConfirmPassword.setVisibility(View.VISIBLE);
            btnAuth.setText("Register");
            tvToggleMode.setText("Already have an account? Login");
        }
        tilUsername.setError(null);
        tilPassword.setError(null);
        tilConfirmPassword.setError(null);
    }

    private void setLoading(boolean isLoading) {
        btnAuth.setEnabled(!isLoading);
        btnGuest.setEnabled(!isLoading);
        btnAuth.setText(isLoading ? "Loading..." : (isLoginMode ? "Login" : "Register"));
    }

    private void showSnackbar(String message) {
        Snackbar.make(findViewById(android.R.id.content), message, Snackbar.LENGTH_LONG).show();
    }

    private void navigateToMain() {
        Intent intent = new Intent(LoginActivity.this, MainActivity.class);
        startActivity(intent);
        finish();
    }
} 