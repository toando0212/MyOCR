package com.example.myocr;

import android.content.Intent;
import android.os.AsyncTask;
import android.os.Bundle;
import android.text.TextUtils;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import androidx.appcompat.app.AppCompatActivity;
import android.app.AlertDialog;
import org.json.JSONObject;
import java.io.OutputStream;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;

public class LoginActivity extends AppCompatActivity {
    private EditText etUsername, etPassword, etConfirmPassword;
    private TextView tvError, tvToggleMode;
    private Button btnAuth;
    private boolean isLoginMode = true;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_login);

        etUsername = findViewById(R.id.etUsername);
        etPassword = findViewById(R.id.etPassword);
        etConfirmPassword = findViewById(R.id.etConfirmPassword);
        tvError = findViewById(R.id.tvError);
        btnAuth = findViewById(R.id.btnAuth);
        tvToggleMode = findViewById(R.id.tvToggleMode);

        updateMode();

        btnAuth.setOnClickListener(v -> {
            String username = etUsername.getText().toString().trim();
            String password = etPassword.getText().toString().trim();
            if (username.length() < 4) {
                showError("Username must be at least 4 characters");
                return;
            }
            if (password.length() < 6) {
                showError("Password must be at least 6 characters");
                return;
            }
            if (isLoginMode) {
                new LoginTask().execute(username, password);
            } else {
                String confirmPassword = etConfirmPassword.getText().toString().trim();
                if (!password.equals(confirmPassword)) {
                    showError("Passwords do not match");
                    return;
                }
                new RegisterTask().execute(username, password);
            }
        });

        tvToggleMode.setOnClickListener(v -> {
            isLoginMode = !isLoginMode;
            updateMode();
        });
    }

    private class RegisterTask extends AsyncTask<String, Void, String> {
        @Override
        protected String doInBackground(String... params) {
            String username = params[0];
            String password = params[1];
            try {
                URL url = new URL("http://192.168.1.132:5000/register"); // Use 10.0.2.2 for Android emulator
                HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                conn.setRequestMethod("POST");
                conn.setRequestProperty("Content-Type", "application/json; utf-8");
                conn.setRequestProperty("Accept", "application/json");
                conn.setDoOutput(true);
                JSONObject jsonInput = new JSONObject();
                jsonInput.put("username", username);
                jsonInput.put("password", password);
                try(OutputStream os = conn.getOutputStream()) {
                    byte[] input = jsonInput.toString().getBytes("utf-8");
                    os.write(input, 0, input.length);
                }
                int code = conn.getResponseCode();
                BufferedReader br = new BufferedReader(new InputStreamReader(
                        code >= 400 ? conn.getErrorStream() : conn.getInputStream(), "utf-8"));
                StringBuilder response = new StringBuilder();
                String responseLine;
                while ((responseLine = br.readLine()) != null) {
                    response.append(responseLine.trim());
                }
                return code + ":" + response.toString();
            } catch (Exception e) {
                return "error:" + e.getMessage();
            }
        }
        @Override
        protected void onPostExecute(String result) {
            if (result.startsWith("error:")) {
                showError("Network error: " + result.substring(6));
                return;
            }
            String[] parts = result.split(":", 2);
            int code = Integer.parseInt(parts[0]);
            String body = parts.length > 1 ? parts[1] : "";
            if (code == 201) {
                new AlertDialog.Builder(LoginActivity.this)
                        .setTitle("Success")
                        .setMessage("Registration successful! You can now log in.")
                        .setPositiveButton("OK", null)
                        .show();
                tvError.setVisibility(View.GONE);
            } else {
                try {
                    JSONObject obj = new JSONObject(body);
                    showError(obj.optString("error", "Registration failed"));
                } catch (Exception e) {
                    showError("Registration failed");
                }
            }
        }
    }

    private class LoginTask extends AsyncTask<String, Void, String> {
        @Override
        protected String doInBackground(String... params) {
            String username = params[0];
            String password = params[1];
            try {
                URL url = new URL("http://192.168.1.132:5000/login");
                HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                conn.setRequestMethod("POST");
                conn.setRequestProperty("Content-Type", "application/json; utf-8");
                conn.setRequestProperty("Accept", "application/json");
                conn.setDoOutput(true);
                JSONObject jsonInput = new JSONObject();
                jsonInput.put("username", username);
                jsonInput.put("password", password);
                try(OutputStream os = conn.getOutputStream()) {
                    byte[] input = jsonInput.toString().getBytes("utf-8");
                    os.write(input, 0, input.length);
                }
                int code = conn.getResponseCode();
                BufferedReader br = new BufferedReader(new InputStreamReader(
                        code >= 400 ? conn.getErrorStream() : conn.getInputStream(), "utf-8"));
                StringBuilder response = new StringBuilder();
                String responseLine;
                while ((responseLine = br.readLine()) != null) {
                    response.append(responseLine.trim());
                }
                return code + ":" + response.toString();
            } catch (Exception e) {
                return "error:" + e.getMessage();
            }
        }
        @Override
        protected void onPostExecute(String result) {
            if (result.startsWith("error:")) {
                showError("Network error: " + result.substring(6));
                return;
            }
            String[] parts = result.split(":", 2);
            int code = Integer.parseInt(parts[0]);
            String body = parts.length > 1 ? parts[1] : "";
            if (code == 200) {
                // Login success, go to MainActivity
                Intent intent = new Intent(LoginActivity.this, MainActivity.class);
                startActivity(intent);
                finish();
            } else {
                try {
                    JSONObject obj = new JSONObject(body);
                    showError(obj.optString("error", "Login failed"));
                } catch (Exception e) {
                    showError("Login failed");
                }
            }
        }
    }

    private void showError(String msg) {
        tvError.setText(msg);
        tvError.setVisibility(View.VISIBLE);
    }

    private void updateMode() {
        if (isLoginMode) {
            etConfirmPassword.setVisibility(View.GONE);
            btnAuth.setText("Login");
            tvToggleMode.setText("Don't have an account? Register");
        } else {
            etConfirmPassword.setVisibility(View.VISIBLE);
            btnAuth.setText("Register");
            tvToggleMode.setText("Already have an account? Login");
        }
        tvError.setVisibility(View.GONE);
    }
} 