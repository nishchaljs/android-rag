package io.datamachines.faiss;

public class LlamaCppWrapper {
    static {
        System.loadLibrary("llama"); // Load libllama.so
    }

    public native boolean loadModel(String modelPath); // Load model from file path

    public native String runInference(String prompt); // Run inference with a prompt

    public native void freeModel(); // Free resources when done
}
