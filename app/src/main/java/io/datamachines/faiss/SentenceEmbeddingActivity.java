package io.datamachines.faiss;

import android.Manifest;
import android.content.ContentUris;
import android.content.Intent;
import android.database.Cursor;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.provider.Settings;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import com.tom_roush.pdfbox.pdmodel.PDDocument;
import com.tom_roush.pdfbox.text.PDFTextStripper;
import com.tom_roush.pdfbox.pdmodel.font.encoding.GlyphList;
import com.tom_roush.pdfbox.android.PDFBoxResourceLoader;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.Reader;
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

import faiss.R;

public class SentenceEmbeddingActivity extends AppCompatActivity {
    private static final int PICK_PDF_FILE = 1;

    private static final String TAG = "SentenceEmbedding";
    private static final String MODEL_FILENAME = "all-minilm-l6-v2/model.onnx";
    private static final String TOKENIZER_FILENAME = "all-minilm-l6-v2/tokenizer.json";

    private EditText editTextInput, editTextSearchQuery;
    private Button btnGenerateEmbedding, btnUploadPdf, btnSearchIndex, btnUseAllPdfs;
    private Button btnCompressEmbedding;
    private TextView tvResults;
    private ProgressBar progressBar;

    private SentenceEmbeddingWrapper sentenceEmbeddingWrapper;
    private float[] currentEmbedding;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        requestPermissions();
        setContentView(R.layout.activity_sentence_embedding);

        PDFBoxResourceLoader.init(getApplicationContext());

        // Initialize UI components
        editTextInput = findViewById(R.id.editTextInput);
        editTextSearchQuery = findViewById(R.id.editTextSearchQuery);
        btnGenerateEmbedding = findViewById(R.id.btnGenerateEmbedding);
        btnCompressEmbedding = findViewById(R.id.btnCompressEmbedding);
        btnSearchIndex = findViewById(R.id.btnSearchIndex);
        tvResults = findViewById(R.id.tvResults);
        progressBar = findViewById(R.id.progressBar);
        btnUploadPdf = findViewById(R.id.btnUploadPdf);
        btnUseAllPdfs = findViewById(R.id.btnUseAllPdfs);

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

        btnUploadPdf.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                selectPdfFile();
            }
        });

        btnSearchIndex.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                searchIndex();
            }
        });

        btnUseAllPdfs.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                useAllPdfsOnDevice();
            }
        });
    }

    private void requestPermissions() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            if (!Environment.isExternalStorageManager()) {
                Intent intent = new Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION);
                intent.setData(Uri.parse("package:" + getPackageName()));
                startActivity(intent);
            }
        } else {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.READ_EXTERNAL_STORAGE, Manifest.permission.WRITE_EXTERNAL_STORAGE},
                    1);
        }
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
                //
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

    private void selectPdfFile() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("application/pdf");
        intent.putExtra(Intent.EXTRA_LOCAL_ONLY, true);
        intent.putExtra(Intent.EXTRA_ALLOW_MULTIPLE, true);
        startActivityForResult(intent, PICK_PDF_FILE);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == PICK_PDF_FILE && resultCode == RESULT_OK && data != null) {
            progressBar.setVisibility(View.VISIBLE);
            List<Uri> uriList = new ArrayList<>();
            if (data.getClipData() != null) {
                int fileCount = data.getClipData().getItemCount();
                for (int i = 0; i < fileCount; i++) {
                    uriList.add(data.getClipData().getItemAt(i).getUri());
                }
            } else if (data.getData() != null) {
                uriList.add(data.getData());
            }
            processPdf(uriList.toArray(new Uri[0]));
        }
    }

    private InputStream getAssetFile(String filename) throws IOException {
        return getAssets().open(filename);
    }

    private void processPdf(Uri[] pdfUris) {
        int chunk_size = 256;
        tvResults.setText("Processing PDF...");

        new Thread(() -> {
            try{
                List<ChunkMetadata> allMetadata = new ArrayList<>();
                List<float[]> allEmbeddings = new ArrayList<>();

                for (Uri pdfUri : pdfUris) {
                    try (InputStream inputStream = getContentResolver().openInputStream(pdfUri)) {
                        PDDocument document = PDDocument.load(inputStream);
                        PDFTextStripper pdfStripper = new PDFTextStripper();
//                        String textContent = pdfStripper.getText(document);
                        int totalPages = document.getNumberOfPages();
                        for (int pageNumber = 1; pageNumber <= totalPages; pageNumber++) {
                            pdfStripper.setStartPage(pageNumber);
                            pdfStripper.setEndPage(pageNumber);
                            String pageText = pdfStripper.getText(document);

                            // Step 1: Split text into sentences with offsets
                            List<Map.Entry<String, Integer>> sentencesWithOffsets =
                                    splitIntoSentencesWithOffsets(pageText);

                            // Step 2: Generate sentence embeddings once
                            List<String> sentences = sentencesWithOffsets.stream()
                                    .map(Map.Entry::getKey).collect(Collectors.toList());
                            List<float[]> sentenceEmbeddings =
                                    generateSentenceEmbeddings(sentences);

                            // Step 3: Create semantic chunks with offsets using precomputed embeddings
                            double threshold = 0.70; // Adjust threshold based on experimentation
                            List<Map.Entry<List<String>, Map.Entry<Integer, Integer>>> semanticChunks =
                                    createSemanticChunksWithOffsets(sentencesWithOffsets, sentenceEmbeddings, threshold);

                            // Step 4: Process each chunk
                            for (Map.Entry<List<String>, Map.Entry<Integer, Integer>> chunkEntry : semanticChunks) {
                                List<String> chunkSentences = chunkEntry.getKey();
                                Map.Entry<Integer, Integer> offsets = chunkEntry.getValue();
                                int startIndex = offsets.getKey();
                                int endIndex = offsets.getValue();

                                float[] chunkEmbedding = aggregateEmbeddings(sentenceEmbeddings.subList(startIndex, endIndex));

                                int startOffset = sentencesWithOffsets.get(startIndex).getValue();
                                int endOffset = sentencesWithOffsets.get(endIndex - 1).getValue()
                                        + sentencesWithOffsets.get(endIndex - 1).getKey().length();

                                allEmbeddings.add(chunkEmbedding);
                                allMetadata.add(new ChunkMetadata(pdfUri.toString(), pageNumber, startOffset, endOffset));
                            }

                            // Split text into chunks
                            // List<String> chunks = splitIntoChunks(pageText, chunk_size);
                            // Generate embeddings for each chunk
//                            int i = 0;
//                            for (String chunk : chunks) {
//                                allEmbeddings.add(sentenceEmbeddingWrapper.encodeSync(chunk));
//                                allMetadata.add(new ChunkMetadata(pdfUri.toString(),
//                                        pageNumber,
//                                        i * chunk_size,
//                                        Math.min((i + 1) * chunk_size, pageText.length())));
//                                i++;
//                            }
                        }

                        document.close();
                    }
                }
                // Compress embeddings using PQ and store them
                compressAndStoreEmbeddings(allEmbeddings);
                saveMetadataToDisk(allMetadata);

//                runOnUiThread(() -> {
//                    tvResults.setText("Embeddings generated for PDF content:\n" +
//                            "Total Chunks: " + chunks.size() + "\n" +
//                            "First Chunk Embedding Size: " + embeddings.get(0).length + " dimensions");
//                    progressBar.setVisibility(View.INVISIBLE);
//                    btnCompressEmbedding.setEnabled(true);
//                });
            } catch (Exception e) {
                Log.e(TAG, "Error processing PDF: " + e.getMessage(), e);
                runOnUiThread(() -> {
                    tvResults.setText("Error processing PDF.");
                    progressBar.setVisibility(View.INVISIBLE);
                });
            }
        }).start();
    }

    private float[] aggregateEmbeddings(List<float[]> chunkEmbeddings) {
        int dimension = chunkEmbeddings.get(0).length; // Get the embedding dimension
        float[] aggregatedEmbedding = new float[dimension];

        // Sum up all embeddings in the chunk
        for (float[] embedding : chunkEmbeddings) {
            for (int i = 0; i < dimension; i++) {
                aggregatedEmbedding[i] += embedding[i];
            }
        }

        // Compute the average by dividing each element by the number of embeddings
        for (int i = 0; i < dimension; i++) {
            aggregatedEmbedding[i] /= chunkEmbeddings.size();
        }

        return aggregatedEmbedding;
    }


    private List<Map.Entry<String, Integer>> splitIntoSentencesWithOffsets(String text) {
        List<Map.Entry<String, Integer>> sentencesWithOffsets = new ArrayList<>();
        String[] sentences = text.split("(?<=[.!?])\\s+"); // Split at sentence boundaries

        int currentOffset = 0;
        for (String sentence : sentences) {
            sentencesWithOffsets.add(new HashMap.SimpleEntry<>(sentence, currentOffset));
            currentOffset += sentence.length() + 1; // Account for space or punctuation
        }

        return sentencesWithOffsets;
    }


    private List<float[]> generateSentenceEmbeddings(List<String> sentences) {
        List<float[]> embeddings = new ArrayList<>();
        for (String sentence : sentences) {
            embeddings.add(sentenceEmbeddingWrapper.encodeSync(sentence)); // Compute embedding once
        }
        return embeddings;
    }

    private List<Map.Entry<List<String>, Map.Entry<Integer, Integer>>> createSemanticChunksWithOffsets(
            List<Map.Entry<String, Integer>> sentencesWithOffsets,
            List<float[]> embeddings,
            double threshold) {

        List<Map.Entry<List<String>, Map.Entry<Integer, Integer>>> chunks = new ArrayList<>();
        List<String> currentChunk = new ArrayList<>();
        int chunkStartOffset = -1;
        int chunkStartIndex = -1;

        for (int i = 0; i < sentencesWithOffsets.size(); i++) {
            Map.Entry<String, Integer> sentenceEntry = sentencesWithOffsets.get(i);
            String sentence = sentenceEntry.getKey();
            int offset = sentenceEntry.getValue();

            if (chunkStartOffset == -1) {
                chunkStartOffset = offset; // Initialize chunk start offset
                chunkStartIndex = i; // Initialize chunk start index
            }

            currentChunk.add(sentence);

            if (i < sentencesWithOffsets.size() - 1) {
                double similarity = calculateCosineSimilarity(embeddings.get(i), embeddings.get(i + 1));
                if (similarity < threshold && currentChunk.size()>=2) { // Break chunk if similarity is below the threshold
                    int chunkEndOffset = offset + sentence.length();
                    chunks.add(new HashMap.SimpleEntry<>(new ArrayList<>(currentChunk),
                            new HashMap.SimpleEntry<>(chunkStartIndex, i+1)));
                    currentChunk.clear();
                    chunkStartOffset = -1; // Reset for next chunk
                    chunkStartIndex = -1;
                }
            }
        }

        if (!currentChunk.isEmpty()) {
            int lastSentenceIndex = sentencesWithOffsets.size() - 1;
            int chunkEndOffset = sentencesWithOffsets.get(lastSentenceIndex).getValue()
                    + sentencesWithOffsets.get(lastSentenceIndex).getKey().length();
            chunks.add(new HashMap.SimpleEntry<>(new ArrayList<>(currentChunk),
                    new HashMap.SimpleEntry<>(chunkStartIndex, lastSentenceIndex+1)));
        }

        // Merge small chunks with adjacent ones
        chunks = mergeSmallChunks(chunks, 2);

        return chunks;
    }

    private List<Map.Entry<List<String>, Map.Entry<Integer, Integer>>> mergeSmallChunks(
            List<Map.Entry<List<String>, Map.Entry<Integer, Integer>>> chunks,
            int minChunkSize) {

        List<Map.Entry<List<String>, Map.Entry<Integer, Integer>>> mergedChunks = new ArrayList<>();

        for (int i = 0; i < chunks.size(); i++) {
            Map.Entry<List<String>, Map.Entry<Integer, Integer>> currentChunk = chunks.get(i);

            if (currentChunk.getKey().size() < minChunkSize && i < chunks.size() - 1) {
                // Merge with next chunk
                Map.Entry<List<String>, Map.Entry<Integer, Integer>> nextChunk = chunks.get(i + 1);

                List<String> mergedSentences = new ArrayList<>(currentChunk.getKey());
                mergedSentences.addAll(nextChunk.getKey());

                int startOffset = currentChunk.getValue().getKey();
                int endOffset = nextChunk.getValue().getValue();

                mergedChunks.add(new HashMap.SimpleEntry<>(mergedSentences,
                        new HashMap.SimpleEntry<>(startOffset, endOffset)));

                i++; // Skip next chunk since it has been merged
            } else {
                mergedChunks.add(currentChunk);
            }
        }

        return mergedChunks;
    }




    private double calculateCosineSimilarity(float[] embedding1, float[] embedding2) {
        double dotProduct = 0.0;
        double normA = 0.0;
        double normB = 0.0;

        for (int i = 0; i < embedding1.length; i++) {
            dotProduct += embedding1[i] * embedding2[i];
            normA += Math.pow(embedding1[i], 2);
            normB += Math.pow(embedding2[i], 2);
        }

        return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    }



    private void useAllPdfsOnDevice() {
        progressBar.setVisibility(View.VISIBLE);
        tvResults.setText("Scanning device for PDFs...");

        new Thread(() -> {
            try {
                List<Uri> pdfUris = new ArrayList<>();

                // Query MediaStore for all PDF files
                String[] projection = {MediaStore.Files.FileColumns._ID};
                String selection = MediaStore.Files.FileColumns.MIME_TYPE + "=?";
                String[] selectionArgs = {"application/pdf"};

                Uri queryUri = MediaStore.Files.getContentUri("external");
                Cursor cursor = getContentResolver().query(queryUri, projection, selection, selectionArgs, null);

                if (cursor != null) {
                    while (cursor.moveToNext()) {
                        long id = cursor.getLong(cursor.getColumnIndexOrThrow(MediaStore.Files.FileColumns._ID));
                        Uri contentUri = ContentUris.withAppendedId(queryUri, id);
                        pdfUris.add(contentUri);
                    }
                    cursor.close();
                }

                if (pdfUris.isEmpty()) {
                    runOnUiThread(() -> {
                        tvResults.setText("No PDFs found on device.");
                        progressBar.setVisibility(View.INVISIBLE);
                    });
                    return;
                }

                processPdf(pdfUris.toArray(new Uri[0])); // Process all found PDFs

            } catch (Exception e) {
                Log.e(TAG, "Error scanning device for PDFs: " + e.getMessage(), e);
                runOnUiThread(() -> {
                    tvResults.setText("Error scanning device for PDFs.");
                    progressBar.setVisibility(View.INVISIBLE);
                });
            }
        }).start();
    }


    private void saveMetadataToDisk(List<ChunkMetadata> metadataList) {
        File directory = new File(getExternalFilesDir(null), "faiss_data");
        if (!directory.exists()) {
            boolean success = directory.mkdirs();
            Log.d(TAG, "Directory creation result: " + success);
        }

        File metadataFile = new File(directory, "chunk_metadata.json");

        try (FileOutputStream fos = new FileOutputStream(metadataFile)) {
            Gson gson = new Gson();
            String jsonString = gson.toJson(metadataList); // Convert metadata list to JSON string
            fos.write(jsonString.getBytes());
            Log.d(TAG, "Metadata saved to disk at " + metadataFile.getAbsolutePath());
        } catch (IOException e) {
            Log.e(TAG, "Error saving metadata to disk: " + e.getMessage(), e);
        }
    }

    private String extractTextFromPdf(Uri pdfFilePath, int pageNumber, int startOffset, int endOffset) {
        try (InputStream inputStream = getContentResolver().openInputStream(pdfFilePath)) {
            PDDocument document = PDDocument.load(inputStream);
            PDFTextStripper pdfStripper = new PDFTextStripper();
            // Set the range to extract only the specified page
            pdfStripper.setStartPage(pageNumber);
            pdfStripper.setEndPage(pageNumber);

            String pageText = pdfStripper.getText(document);
            document.close();

            return pageText.substring(startOffset, Math.min(endOffset, pageText.length()));
        } catch (IOException e) {
            Log.e(TAG, "Error extracting text from PDF: " + e.getMessage(), e);
            return "Error extracting text.";
        }
    }


    private List<ChunkMetadata> loadMetadataFromDisk() {
        File directory = new File(getExternalFilesDir(null), "faiss_data");
        File metadataFile = new File(directory, "chunk_metadata.json");

        if (!metadataFile.exists()) {
            Log.e(TAG, "Metadata file not found at " + metadataFile.getAbsolutePath());
            return null;
        }

        try (InputStream is = new FileInputStream(metadataFile)) {
            Gson gson = new Gson();
            Reader reader = new InputStreamReader(is);
            Type listType = new TypeToken<List<ChunkMetadata>>() {}.getType();
            return gson.fromJson(reader, listType); // Deserialize JSON string into a list of ChunkMetadata
        } catch (IOException e) {
            Log.e(TAG, "Error loading metadata from disk: " + e.getMessage(), e);
            return null;
        }
    }

    private List<String> splitIntoChunks(String text, int chunkSize) {
        List<String> chunks = new ArrayList<>();
        int length = text.length();
        for (int i = 0; i < length; i += chunkSize) {
            chunks.add(text.substring(i, Math.min(length, i + chunkSize)));
        }
        return chunks;
    }

    private void compressAndStoreEmbeddings(List<float[]> embeddings) {
        tvResults.setText("Compressing embeddings...");

        // Convert List<float[]> to float[][] for JNI compatibility
        int numEmbeddings = embeddings.size();
        int dimension = embeddings.get(0).length;
        float[][] embeddingArray = new float[numEmbeddings][dimension];
        for (int i = 0; i < numEmbeddings; i++) {
            embeddingArray[i] = embeddings.get(i);
        }

        File directory = new File(getExternalFilesDir(null), "faiss_data");
        if (!directory.exists()) {
            boolean success = directory.mkdirs();
            Log.d(TAG, "Directory creation result: " + success);
        }
        String storagePath = directory.getAbsolutePath();

        new Thread(() -> {
            try {
                String result = compressEmbeddingsWithPQ(embeddingArray, 300 /* Training vectors */, storagePath);

                runOnUiThread(() -> {
                    tvResults.setText(result);
                    progressBar.setVisibility(View.INVISIBLE);
                });
            } catch (Exception e) {
                Log.e(TAG, "Error compressing embeddings: " + e.getMessage(), e);

                runOnUiThread(() -> {
                    tvResults.setText("Error compressing embeddings.");
                    progressBar.setVisibility(View.INVISIBLE);
                });
            }
        }).start();
    }

    private void searchIndex() {
        String query = editTextSearchQuery.getText().toString().trim();

        if (query.isEmpty()) {
            Toast.makeText(this, "Please enter a search query", Toast.LENGTH_SHORT).show();
            return;
        }

        progressBar.setVisibility(View.VISIBLE);

        new Thread(() -> {
            try {
                float[] queryEmbedding = sentenceEmbeddingWrapper.encodeSync(query); // Generate query embedding

                File directory = new File(getExternalFilesDir(null), "faiss_data");
                String indexPath = directory.getAbsolutePath() + "/embeddings_pq.index";

                String results = searchIndexNative(queryEmbedding,15, indexPath); // Search top-5 results
                // Deserialize search results
                Gson gson = new Gson();
                Type resultType = new TypeToken<List<Integer>>() {}.getType();
                List<Integer> topKIndices = gson.fromJson(results, resultType);

                // Map results to text using metadata
                List<ChunkMetadata> metadataList = loadMetadataFromDisk();
                if (metadataList == null) throw new Exception("Failed to load metadata.");

//                StringBuilder resultBuilder = new StringBuilder("Search Results:\n");
//                for (int i = 0; i < topKIndices.size(); i++) {
//                    int index = topKIndices.get(i);
//                    if (index >= 0 && index < metadataList.size()) {
//                        ChunkMetadata metadata = metadataList.get(index);
//
//                        String extractedText = extractTextFromPdf(
//                                Uri.parse(metadata.getPdfFilePath()),
//                                metadata.getPageNumber(),
//                                metadata.getStartOffset(),
//                                metadata.getEndOffset()
//                        );
//
//                        resultBuilder.append("Rank ").append(i + 1).append(": ");
//                        resultBuilder.append("PDF Path=").append(metadata.getPdfFilePath()).append(", ");
//                        resultBuilder.append("Extracted Text: ").append(extractedText).append("\n\n");
//                    }
//                }
                Map<Integer, ChunkMetadata> metadataMap = new HashMap<>();
                for (int i = 0; i < metadataList.size(); i++) {
                    metadataMap.put(i, metadataList.get(i));
                }

                ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
                AtomicInteger rankCounter = new AtomicInteger(1);
                // Parallel text extraction
                List<CompletableFuture<String>> futures = topKIndices.stream()
                        .filter(metadataMap::containsKey) // Pre-filter valid indices
                        .map(index -> CompletableFuture.supplyAsync(() -> {
                            ChunkMetadata metadata = metadataMap.get(index);
                            Uri pdfUri = Uri.parse(metadata.getPdfFilePath());
                            String extractedText = extractTextFromPdf(
                                    pdfUri,
                                    metadata.getPageNumber(),
                                    metadata.getStartOffset(),
                                    metadata.getEndOffset()
                            );
                            return String.format("Rank %d: PDF Path=%s, Extracted Text: %s\n\n",
                                    rankCounter.getAndIncrement(), metadata.getPdfFilePath(), extractedText);
                        }, executor))
                        .collect(Collectors.toList());

                // Collect results
                CompletableFuture.allOf(futures.toArray(new CompletableFuture[0])).join(); // Wait for all tasks to finish

                String resultText = futures.stream()
                        .map(CompletableFuture::join)
                        .collect(Collectors.joining());

                executor.shutdown();


                runOnUiThread(() -> {
                    tvResults.setText("Search Results:\n" + resultText); // Display results
                    progressBar.setVisibility(View.INVISIBLE);
                });
            } catch (Exception e) {
                Log.e(TAG, "Error searching index: " + e.getMessage(), e);
                runOnUiThread(() -> progressBar.setVisibility(View.INVISIBLE));
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
    private native String compressEmbeddingsWithPQ(float[][] embeddings, int numTrainingVectors,
                                                   String storagePath);

    private native String searchIndexNative(float[] queryEmbedding, int topK, String indexPath);

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