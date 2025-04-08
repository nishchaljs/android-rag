package io.datamachines.faiss;

import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import java.io.File;
import java.io.InputStream;

import faiss.R;

public class MainActivity extends AppCompatActivity {

    // Used to load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("faiss");
    }

    Button btnRunTest;
    TextView tvResults;
    ProgressBar progressBar;
    String resultData;
    Button btnOpenEmbeddingActivity;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Initialize UI components
        tvResults = findViewById(R.id.sample_text);
        tvResults.setText("Product Quantization Demo");

        progressBar = findViewById(R.id.progressBar); // Add a ProgressBar to your layout
        progressBar.setVisibility(View.INVISIBLE);

        btnRunTest = findViewById(R.id.button);
        btnRunTest.setText("Run PQ Demo");
        btnRunTest.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                runProductQuantizationDemo();
            }
        });

        // Add new button for sentence embedding activity
        btnOpenEmbeddingActivity = findViewById(R.id.btnOpenEmbeddingActivity);
        btnOpenEmbeddingActivity.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(MainActivity.this, SentenceEmbeddingActivity.class);
                startActivity(intent);
            }
        });

        // In your Activity
        LlamaCppWrapper llama = new LlamaCppWrapper();

        // Load model from external storage
        String modelPath = new File(getExternalFilesDir(null), "qwen2.5-0.5b-instruct-q4_k_m.gguf").getAbsolutePath();
        if (llama.loadModel(modelPath)) {
            String response = llama.runInference("Explain quantum computing");
            Log.d("Model response", response);
            // Update UI with response
        }

// Always free resources
        llama.freeModel();


    }

    void runProductQuantizationDemo() {
        // Create directory for storing the index
        File directory = new File(getExternalFilesDir(null), "faiss_data");
        if (!directory.exists()) {
            boolean success = directory.mkdirs();
            Log.d("Storage_path", "Directory creation result: " + success);
        }
        String storagePath = directory.getAbsolutePath();
        Log.d("Storage_path", "Using path: " + storagePath);

        // Disable button and show progress
        btnRunTest.setEnabled(false);
        progressBar.setVisibility(View.VISIBLE);
        tvResults.setText("Running Product Quantization Demo...\nThis may take a minute.");

        // Run the native code in a background thread
        new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    // Call native method
                    resultData = stringFromJNI(0, storagePath);

                    // Update UI on the main thread
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            updateResults();
                        }
                    });
                } catch (Exception e) {
                    final String errorMsg = "Error: " + e.getMessage();
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            Toast.makeText(MainActivity.this, errorMsg, Toast.LENGTH_LONG).show();
                            tvResults.setText("Error occurred. Check logs.");
                            btnRunTest.setEnabled(true);
                            progressBar.setVisibility(View.INVISIBLE);
                        }
                    });
                    Log.e("PQ_Demo", "Error running demo", e);
                }
            }
        }).start();
    }

    void updateResults() {
        tvResults.setText(resultData);
        btnRunTest.setEnabled(true);
        progressBar.setVisibility(View.INVISIBLE);
    }

    // Native method declaration
    public static native String stringFromJNI(int a, String storage_path);
}
