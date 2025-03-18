package io.datamachines.faiss;

import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

import faiss.R;

public class SentenceEmbeddingActivity extends AppCompatActivity {

    private static final String TAG = "SentenceEmbedding";
    private static final String MODEL_FILENAME = "all-minilm-l6-v2/model.onnx";
    private static final String TOKENIZER_FILENAME = "all-minilm-l6-v2/tokenizer.json";

    private EditText editTextInput;
    private Button btnGenerateEmbedding;
    private Button btnCompressEmbedding;
    private TextView tvResults;
    private ProgressBar progressBar;

    private SentenceEmbeddingWrapper sentenceEmbeddingWrapper;
    private float[] currentEmbedding;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_sentence_embedding);

        // Initialize UI components
        editTextInput = findViewById(R.id.editTextInput);
        btnGenerateEmbedding = findViewById(R.id.btnGenerateEmbedding);
        btnCompressEmbedding = findViewById(R.id.btnCompressEmbedding);
        tvResults = findViewById(R.id.tvResults);
        progressBar = findViewById(R.id.progressBar);

        // Initialize sentence embedding
        initializeSentenceEmbedding();

        // Set up click listeners
        btnGenerateEmbedding.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                generateEmbedding();
            }
        });

        btnCompressEmbedding.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                compressEmbedding();
            }
        });

        // Initially disable compression button
        btnCompressEmbedding.setEnabled(false);
    }

    private void initializeSentenceEmbedding() {
        progressBar.setVisibility(View.VISIBLE);
        tvResults.setText("Initializing embedding model...");

        new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    // Copy model files from assets to internal storage
                    File modelFile = copyAssetToInternalStorage(MODEL_FILENAME);
                    File tokenizerFile = copyAssetToInternalStorage(TOKENIZER_FILENAME);

                    // Initialize SentenceEmbedding
                    sentenceEmbeddingWrapper = new SentenceEmbeddingWrapper();
                    sentenceEmbeddingWrapper.initSync(
                            modelFile.getAbsolutePath(),
                            getBytesFromFile(tokenizerFile),
                            true,
                            "sentence_embedding",
                            true);

                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            tvResults.setText("Model initialized successfully!\nReady to generate embeddings.");
                            progressBar.setVisibility(View.INVISIBLE);
                            btnGenerateEmbedding.setEnabled(true);
                        }
                    });
                } catch (Exception e) {
                    final String errorMessage = "Error initializing model: " + e.getMessage();
                    Log.e(TAG, errorMessage, e);

                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            tvResults.setText(errorMessage);
                            progressBar.setVisibility(View.INVISIBLE);
                            Toast.makeText(SentenceEmbeddingActivity.this,
                                    "Failed to initialize model", Toast.LENGTH_LONG).show();
                        }
                    });
                }
            }
        }).start();
    }

    private void generateEmbedding() {
        String text = editTextInput.getText().toString().trim();

        if (text.isEmpty()) {
            Toast.makeText(this, "Please enter some text", Toast.LENGTH_SHORT).show();
            return;
        }

        progressBar.setVisibility(View.VISIBLE);
        btnGenerateEmbedding.setEnabled(false);
        tvResults.setText("Generating embedding...");

        new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    // Generate embedding
                    currentEmbedding = sentenceEmbeddingWrapper.encodeSync(text);

                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            StringBuilder resultBuilder = new StringBuilder();
                            resultBuilder.append("Embedding generated successfully!\n\n");
                            resultBuilder.append("Text: ").append(text).append("\n");
                            resultBuilder.append("Embedding size: ").append(currentEmbedding.length).append(" dimensions\n");
                            resultBuilder.append("First 5 values: ");

                            for (int i = 0; i < Math.min(5, currentEmbedding.length); i++) {
                                resultBuilder.append(String.format("%.4f", currentEmbedding[i]));
                                if (i < 4) resultBuilder.append(", ");
                            }

                            resultBuilder.append("\n\nOriginal size: ")
                                    .append(String.format("%.2f", currentEmbedding.length * 4 / 1024.0f))
                                    .append(" KB");

                            tvResults.setText(resultBuilder.toString());
                            progressBar.setVisibility(View.INVISIBLE);
                            btnGenerateEmbedding.setEnabled(true);
                            btnCompressEmbedding.setEnabled(true);
                        }
                    });
                } catch (Exception e) {
                    final String errorMessage = "Error generating embedding: " + e.getMessage();
                    Log.e(TAG, errorMessage, e);

                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            tvResults.setText(errorMessage);
                            progressBar.setVisibility(View.INVISIBLE);
                            btnGenerateEmbedding.setEnabled(true);
                        }
                    });
                }
            }
        }).start();
    }

    private void compressEmbedding() {
        if (currentEmbedding == null) {
            Toast.makeText(this, "No embedding to compress", Toast.LENGTH_SHORT).show();
            return;
        }

        progressBar.setVisibility(View.VISIBLE);
        btnCompressEmbedding.setEnabled(false);
        tvResults.setText(tvResults.getText() + "\n\nCompressing embedding...");

        // Create storage path
        File directory = new File(getExternalFilesDir(null), "faiss_data");
        if (!directory.exists()) {
            boolean success = directory.mkdirs();
            Log.d(TAG, "Directory creation result: " + success);
        }
        String storagePath = directory.getAbsolutePath();

        new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    // Call native method to compress the embedding
                    String result = compressEmbeddingWithPQ(currentEmbedding, storagePath);

                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            tvResults.setText(tvResults.getText() + "\n\n" + result);
                            progressBar.setVisibility(View.INVISIBLE);
                            btnCompressEmbedding.setEnabled(true);
                        }
                    });
                } catch (Exception e) {
                    final String errorMessage = "Error compressing embedding: " + e.getMessage();
                    Log.e(TAG, errorMessage, e);

                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            tvResults.setText(tvResults.getText() + "\n\n" + errorMessage);
                            progressBar.setVisibility(View.INVISIBLE);
                            btnCompressEmbedding.setEnabled(true);
                        }
                    });
                }
            }
        }).start();
    }

    // Helper method to compress embedding using the native PQ implementation
    private native String compressEmbeddingWithPQ(float[] embedding, String storagePath);

    // Helper method to copy asset file to internal storage
    private File copyAssetToInternalStorage(String assetName) throws Exception {
        // Create the destination file reference
        File outputFile = new File(getFilesDir(), assetName);

        // Ensure parent directories exist
        if (!outputFile.getParentFile().exists()) {
            boolean dirCreated = outputFile.getParentFile().mkdirs();
            Log.d("SentenceEmbedding", "Directory creation result: " + dirCreated);
        }

        // Copy the file
        try (InputStream in = getAssets().open(assetName);
             OutputStream out = new FileOutputStream(outputFile)) {

            byte[] buffer = new byte[1024];
            int read;

            while ((read = in.read(buffer)) != -1) {
                out.write(buffer, 0, read);
            }

            Log.d("SentenceEmbedding", "Successfully copied " + assetName + " to " + outputFile.getAbsolutePath());
        } catch (IOException e) {
            Log.e("SentenceEmbedding", "Failed to copy asset " + assetName, e);
            throw e;
        }

        return outputFile;
    }

    // Helper method to get bytes from file
    private byte[] getBytesFromFile(File file) throws Exception {
        InputStream in = new java.io.FileInputStream(file);
        byte[] bytes = new byte[(int) file.length()];

        try {
            in.read(bytes);
        } finally {
            in.close();
        }

        return bytes;
    }

    static {
        System.loadLibrary("faiss");
    }
}