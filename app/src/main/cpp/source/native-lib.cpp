#include <jni.h>
#include <string>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <sys/time.h>
#include <random>

#include "faiss/IndexIVFPQ.h"
#include "faiss/IndexFlat.h"
#include "faiss/index_io.h"

#include "llama.cpp/include/llama.h"
#include "llama.cpp/ggml/include/ggml.h"
#include "llama.cpp/common/common.h"

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

#include "log.h"
using namespace faiss;

// Global state with thread safety
struct {
    llama_model* model;
    llama_context* ctx;
    llama_sampler* sampler;
    std::vector<llama_chat_message> messages;
    std::mutex mutex;
} llama_state;

llama_sampler* initialize_sampler() {
    llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();
    sampler_params.no_perf                    = true; // disable performance metrics
    llama_sampler* sampler = llama_sampler_chain_init(sampler_params);
    llama_sampler_chain_add(sampler, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.9));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    return sampler;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_io_datamachines_faiss_LlamaCppWrapper_loadModel(JNIEnv* env, jobject, jstring modelPath) {
    std::lock_guard<std::mutex> lock(llama_state.mutex);
    const char* path = env->GetStringUTFChars(modelPath, nullptr);

    llama_model_params model_params = llama_model_default_params();
    model_params.use_mmap           = true;
    model_params.use_mlock          = false;
    model_params.vocab_only = false;
    model_params.kv_overrides = NULL;
//    model_params.tensor_buft_overrides = NULL;

//    model_params.progress_callback = [](float progress, void*) -> bool {
//        LOGI("Loading model: %.2f%%", progress * 100);
//        return true;
//    };

    llama_model* model = llama_model_load_from_file(path, model_params);
    llama_state.model = model;
    env->ReleaseStringUTFChars(modelPath, path);

    if (!llama_state.model) {
        LOGE("Failed to load model");
        return JNI_FALSE;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048;

    llama_context* ctx = llama_init_from_model(llama_state.model, ctx_params);
    llama_state.ctx = ctx;
    if (!llama_state.ctx) {
        LOGE("Failed to create context");
        llama_model_free(llama_state.model);
        llama_state.model = nullptr;
        return JNI_FALSE;
    }

    llama_sampler* sampler = initialize_sampler();
    llama_state.sampler = sampler;
    llama_state.messages.push_back({strdup("system"), strdup("You are a helpful assistant.")});
    //    llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();
//    sampler_params.no_perf = true;
//    llama_state.sampler = llama_sampler_chain_init(sampler_params);
//    llama_sampler_chain_add(llama_state.sampler, llama_sampler_init_min_p(0.05f,1));
//    llama_sampler_chain_add(llama_state.sampler, llama_sampler_init_temp(0.9));
//    llama_sampler_chain_add(llama_state.sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    return JNI_TRUE;
}

extern "C" JNIEXPORT jstring JNICALL
Java_io_datamachines_faiss_LlamaCppWrapper_runInference(JNIEnv* env, jobject, jstring prompt) {
    std::lock_guard<std::mutex> lock(llama_state.mutex);
    if (!llama_state.ctx || !llama_state.model) {
        return env->NewStringUTF("Model not loaded");
    }

    const char* input = env->GetStringUTFChars(prompt, nullptr);
    std::string output;
    std::vector<char> _formattedMessages = std::vector<char>(llama_n_ctx(llama_state.ctx));
    llama_state.messages.push_back({strdup("user"), strdup(input)});
    int newLen = llama_chat_apply_template(NULL, llama_state.messages.data(), llama_state.messages.size(), true,
                                           _formattedMessages.data(), _formattedMessages.size());

    if (newLen > (int)_formattedMessages.size()) {
        // resize the output buffer `_formattedMessages`
        // and re-apply the chat template
        _formattedMessages.resize(newLen);
        newLen = llama_chat_apply_template(NULL, llama_state.messages.data(), llama_state.messages.size(), true,
                                           _formattedMessages.data(), _formattedMessages.size());
    }

    if (newLen < 0) {
        throw std::runtime_error("llama_chat_apply_template() in LLMInference::startCompletion() failed");
    }

    std::string _prompt(_formattedMessages.begin() + 0, _formattedMessages.begin() + newLen);

    llama_batch batch = llama_batch_init(512, 0, 1);
    auto vocab = llama_model_get_vocab(llama_state.model);

    try {
        std::vector<llama_token> tokens = common_tokenize(vocab, _prompt, true, true);

        for (size_t i = 0; i < tokens.size(); i++) {
            batch.token[i] = tokens[i];
            batch.pos[i] = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = (i == tokens.size() - 1);  // only last token requests logits
        }
        batch.n_tokens = tokens.size();
//        int n_seq = 0;
//        for (size_t i = 0; i < tokens.size(); i++) {
//            common_batch_add(batch, tokens[i], static_cast<llama_pos>(i), {n_seq}, (i == tokens.size() - 1));
//        }

        if (llama_decode(llama_state.ctx, batch) != 0) {
            throw std::runtime_error("Initial decoding failed");
        }

        if (!llama_state.sampler) {
            throw std::runtime_error("Sampler is not initialized");
        }

        int n_past = tokens.size();

        for (int i = 0; ; ++i) {
            const float* logits = llama_get_logits(llama_state.ctx);
            if (logits == nullptr) {
                throw std::runtime_error("Logits are null before sampling");
            }
            llama_token new_token = llama_sampler_sample(llama_state.sampler, llama_state.ctx, -1);
            if (llama_vocab_is_eog(llama_model_get_vocab(llama_state.model), new_token)) {
                output += "[EOG]";
                break;
            }

            std::string piece = common_token_to_piece(llama_state.ctx, new_token, true);
            //llama_token_to_piece(vocab, new_token, token_piece, sizeof(token_piece), 1, false);
            output += std::string(piece);

            common_batch_clear(batch); // Reset batch
            batch.token[0] = new_token;
            batch.pos[0] = n_past;
            batch.seq_id[0][0] = 0;
            batch.n_seq_id[0] = 1;
            batch.logits[0] = true;
            batch.n_tokens = 1;

            if (llama_decode(llama_state.ctx, batch) != 0) {
                throw std::runtime_error("Decoding failed");
            }

            n_past+=1;
        }
    } catch (const std::exception& e) {
        output = "Error: " + std::string(e.what());
    }

    llama_batch_free(batch);
    env->ReleaseStringUTFChars(prompt, input);
    return env->NewStringUTF(output.c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_io_datamachines_faiss_LlamaCppWrapper_freeModel(JNIEnv*, jobject) {
    std::lock_guard<std::mutex> lock(llama_state.mutex);
    if (llama_state.ctx) {
        llama_free(llama_state.ctx);
        llama_state.ctx = nullptr;
    }
    if (llama_state.model) {
        llama_model_free(llama_state.model);
        llama_state.model = nullptr;
    }
}


int64_t getCurrentMillTime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return ((int64_t) tv.tv_sec * 1000 + (int64_t) tv.tv_usec / 1000);//毫秒
}
// Generate random vectors for training and testing
void generateRandomVectors(float* vectors, int numVectors, int dimension) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (int i = 0; i < numVectors * dimension; i++) {
        vectors[i] = dis(gen);
    }
}

// Function to randomly sample training vectors
void sampleTrainingVectors(float* allEmbeddings, int numEmbeddings, int dimension, float* trainingVectors, int numTrainingVectors) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, numEmbeddings - 1);

    for (int i = 0; i < numTrainingVectors; i++) {
        int randomIndex = dis(gen);
        for (int j = 0; j < dimension; j++) {
            trainingVectors[i * dimension + j] = allEmbeddings[randomIndex * dimension + j];
        }
    }
}

extern "C" JNIEXPORT jstring

JNICALL stringFromJNI(JNIEnv *env, jclass clazz, jint number, jstring storage_path) {
    std::string result = "";
    const char *cPath = env->GetStringUTFChars(storage_path, nullptr);
    std::string index_path = std::string(cPath) + "/pq.index";
    env->ReleaseStringUTFChars(storage_path, cPath);
    // Product Quantization parameters
    const int dimension = 512;      // Vector dimension
    const int numSubvectors = 64;   // Number of subquantizers (m)
    const int bitsPerSubvector = 8; // Bits per subquantizer (8 = 256 centroids)
    // Timing variables
    int64_t startTime, endTime;

// === Create the PQ index ===
    LOGI("Creating Product Quantization index (d=%d, m=%d, bits=%d)",
         dimension, numSubvectors, bitsPerSubvector);
    faiss::IndexPQ* index = new faiss::IndexPQ(dimension, numSubvectors, bitsPerSubvector);

    // === Generate training vectors ===
    const int numTrainingVectors = 1000;
    LOGI("Generating %d random training vectors", numTrainingVectors);
    float* trainingVectors = new float[numTrainingVectors * dimension];
    generateRandomVectors(trainingVectors, numTrainingVectors, dimension);

    // === Train the PQ index ===
    LOGI("Training PQ index...");
    startTime = getCurrentMillTime();
    index->train(numTrainingVectors, trainingVectors);
    endTime = getCurrentMillTime();
    LOGI("Training completed in %lld ms", endTime - startTime);

    // === Add vectors to the index ===
    const int numAddVectors = 10000;
    LOGI("Adding %d vectors to index...", numAddVectors);
    startTime = getCurrentMillTime();

    // Generate vectors in smaller batches to avoid memory issues
    const int batchSize = 1000;
    float* addVectors = new float[batchSize * dimension];
    int i;
    for (i = 0; i < numAddVectors / batchSize - 1; i++) {
        generateRandomVectors(addVectors, batchSize, dimension);
        index->add(batchSize, addVectors);
        LOGI("Added batch %d/%d, total vectors: %lld",
             i+1, numAddVectors/batchSize, index->ntotal);
    }
    index->add(batchSize, trainingVectors);
    LOGI("Added batch %d/%d, total vectors: %lld",
         i+1, numAddVectors/batchSize, index->ntotal);
    endTime = getCurrentMillTime();
    LOGI("Added %lld vectors in %lld ms", index->ntotal, endTime - startTime);

    // === Save the index ===
    LOGI("Saving index to %s", index_path.c_str());
    startTime = getCurrentMillTime();
    faiss::write_index(index, index_path.c_str());
    endTime = getCurrentMillTime();
    LOGI("Index saved in %lld ms", endTime - startTime);

    // === Search example ===
    const int numQueries = 5;
    const int topK = 10;
    LOGI("Performing %d search queries, retrieving top %d results each", numQueries, topK);

    float* queryVectors = new float[numQueries * dimension];
    generateRandomVectors(queryVectors, numQueries, dimension);

    int64_t* indices = new int64_t[numQueries * topK];
    float* distances = new float[numQueries * topK];

    startTime = getCurrentMillTime();
    index->search(numQueries, queryVectors, topK, distances, indices);
    endTime = getCurrentMillTime();
    LOGI("Search completed in %lld ms", endTime - startTime);

    // Print some sample results
    for (int i = 0; i < std::min(3, numQueries); i++) {
        LOGI("Query %d results:", i);
        for (int j = 0; j < std::min(5, topK); j++) {
            LOGI("  Result %d: ID=%lld, Distance=%f",
                 j, indices[i * topK + j], distances[i * topK + j]);
        }
    }

    // === Calculate memory usage ===
    float originalMemoryMB = (float)(index->ntotal * dimension * sizeof(float)) / (1024 * 1024);
    float compressedMemoryMB = (float)(index->ntotal * numSubvectors) / (1024 * 1024);
    float compressionRatio = originalMemoryMB / compressedMemoryMB;

    LOGI("Memory usage:");
    LOGI("  Original: %.2f MB", originalMemoryMB);
    LOGI("  Compressed: %.2f MB", compressedMemoryMB);
    LOGI("  Compression ratio: %.2f:1", compressionRatio);

    // Prepare result string
    result = "Product Quantization Results:\n";
    result += "- Vectors: " + std::to_string(index->ntotal) + "\n";
    result += "- Dimension: " + std::to_string(dimension) + "\n";
    result += "- Subvectors (m): " + std::to_string(numSubvectors) + "\n";
    result += "- Original size: " + std::to_string(originalMemoryMB) + " MB\n";
    result += "- Compressed: " + std::to_string(compressedMemoryMB) + " MB\n";
    result += "- Compression ratio: " + std::to_string(compressionRatio) + ":1\n";
    result += "- Search time: " + std::to_string(endTime - startTime) + " ms";

    // Clean up
    delete[] trainingVectors;
    delete[] addVectors;
    delete[] queryVectors;
    delete[] indices;
    delete[] distances;
    delete index;

    return env->NewStringUTF(result.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_io_datamachines_faiss_SentenceEmbeddingActivity_compressEmbeddingWithPQ(
        JNIEnv *env, jobject thiz, jfloatArray embedding, jstring storage_path) {

    std::string result = "";
    const char *cPath = env->GetStringUTFChars(storage_path, nullptr);
    std::string index_path = std::string(cPath) + "/embedding_pq.index";
    env->ReleaseStringUTFChars(storage_path, cPath);

    // Get embedding data from Java
    jsize embeddingLength = env->GetArrayLength(embedding);
    jfloat *embeddingData = env->GetFloatArrayElements(embedding, nullptr);

    // Log the embedding size
    LOGI("Received embedding with size: %d", embeddingLength);

    // If the embedding isn't exactly 384 dimensions (from all-MiniLM-L6-v2),
    // we need to pad or truncate
    const int targetDimension = 384;
    float* paddedEmbedding = new float[targetDimension];

    for (int i = 0; i < targetDimension; i++) {
        if (i < embeddingLength) {
            paddedEmbedding[i] = embeddingData[i];
        } else {
            paddedEmbedding[i] = 0.0f; // Pad with zeros if needed
        }
    }

    // Product Quantization parameters
    const int dimension = targetDimension;
    const int numSubvectors = 48;   // For 384 dimensions, 48 subvectors means each is 8-dimensional
    const int bitsPerSubvector = 8; // 8 bits = 256 centroids per subvector

    // Timing variables
    int64_t startTime, endTime;

    // === Create the PQ index ===
    LOGI("Creating Product Quantization index (d=%d, m=%d, bits=%d)",
         dimension, numSubvectors, bitsPerSubvector);
    faiss::IndexPQ* index = new faiss::IndexPQ(dimension, numSubvectors, bitsPerSubvector);

    // === Train the PQ index ===
    // For a single embedding, we don't have enough data to train properly
    // So we'll create synthetic variations for training
    const int numTrainingVectors = 500;
    LOGI("Generating %d training vectors from the embedding", numTrainingVectors);

    float* trainingVectors = new float[numTrainingVectors * dimension];

    // Copy the original embedding and create variations
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f); // Small variations

    for (int i = 0; i < numTrainingVectors; i++) {
        for (int j = 0; j < dimension; j++) {
            // Add small random variations to create synthetic training data
            trainingVectors[i * dimension + j] = paddedEmbedding[j] + dist(gen);
        }
    }

    // Train the index
    LOGI("Training PQ index...");
    startTime = getCurrentMillTime();
    index->train(numTrainingVectors, trainingVectors);
    endTime = getCurrentMillTime();
    LOGI("Training completed in %lld ms", endTime - startTime);

    // Add the original embedding to the index
    LOGI("Adding embedding to index...");
    index->add(1, paddedEmbedding);

    // Save the index
    LOGI("Saving index to %s", index_path.c_str());
    faiss::write_index(index, index_path.c_str());

    // Calculate memory usage
    float originalMemoryKB = (float)(dimension * sizeof(float)) / 1024.0f;
    float compressedMemoryKB = (float)(numSubvectors) / 1024.0f;
    float compressionRatio = originalMemoryKB / compressedMemoryKB;

    LOGI("Memory usage:");
    LOGI("  Original: %.2f KB", originalMemoryKB);
    LOGI("  Compressed: %.2f KB", compressedMemoryKB);
    LOGI("  Compression ratio: %.2f:1", compressionRatio);

    // Prepare result string
    result = "Compression Results:\n";
    result += "- Original size: " + std::to_string(originalMemoryKB) + " KB\n";
    result += "- Compressed size: " + std::to_string(compressedMemoryKB) + " KB\n";
    result += "- Compression ratio: " + std::to_string(compressionRatio) + ":1\n";
    result += "- PQ Parameters: " + std::to_string(numSubvectors) + " subvectors, ";
    result += std::to_string(bitsPerSubvector) + " bits each\n";
    result += "- Index saved to: " + index_path;

    // Clean up
    delete[] paddedEmbedding;
    delete[] trainingVectors;
    env->ReleaseFloatArrayElements(embedding, embeddingData, JNI_ABORT);
    delete index;

    return env->NewStringUTF(result.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_io_datamachines_faiss_SentenceEmbeddingActivity_compressEmbeddingsWithPQ(
        JNIEnv *env, jobject thiz, jobjectArray embeddings,
        jint numTrainingVectors, jstring storage_path) {

    std::string result = "";
    const char *cPath = env->GetStringUTFChars(storage_path, nullptr);
    std::string index_path = std::string(cPath) + "/embeddings_pq.index";
    env->ReleaseStringUTFChars(storage_path, cPath);

    // Get the number of embeddings and their dimension
    jsize numEmbeddings = env->GetArrayLength(embeddings);
    jfloatArray firstEmbedding = (jfloatArray) env->GetObjectArrayElement(embeddings, 0);
    jsize dimension = env->GetArrayLength(firstEmbedding);

    LOGI("Received %d embeddings with dimension %d", numEmbeddings, dimension);

    // Allocate memory for all embeddings
    float* allEmbeddings = new float[numEmbeddings * dimension];
    for (int i = 0; i < numEmbeddings; i++) {
        jfloatArray embedding = (jfloatArray) env->GetObjectArrayElement(embeddings, i);
        jfloat* embeddingData = env->GetFloatArrayElements(embedding, nullptr);

        for (int j = 0; j < dimension; j++) {
            allEmbeddings[i * dimension + j] = embeddingData[j];
        }

        env->ReleaseFloatArrayElements(embedding, embeddingData, JNI_ABORT);
    }

    // === Create the PQ index ===
    const int numSubvectors = 64;   // Number of subquantizers (m)
    const int bitsPerSubvector = 8; // Bits per subquantizer (8 bits = 256 centroids per subvector)
    LOGI("Creating Product Quantization index (d=%d, m=%d, bits=%d)",
         dimension, numSubvectors, bitsPerSubvector);
    faiss::IndexPQ* index = new faiss::IndexPQ(dimension, numSubvectors, bitsPerSubvector);

    // === Train the PQ index ===
    LOGI("Training PQ index with %d vectors...", numTrainingVectors);
    float* trainingVectors = new float[numTrainingVectors * dimension];
    sampleTrainingVectors(allEmbeddings, numEmbeddings, dimension, trainingVectors, numTrainingVectors);

    int64_t startTime = getCurrentMillTime();
    index->train(numTrainingVectors, trainingVectors);
    int64_t endTime = getCurrentMillTime();
    LOGI("Training completed in %lld ms", endTime - startTime);

    delete[] trainingVectors;

    // === Add all embeddings to the index ===
    LOGI("Adding %d embeddings to the index...", numEmbeddings);
    startTime = getCurrentMillTime();
    index->add(numEmbeddings, allEmbeddings);
    endTime = getCurrentMillTime();
    LOGI("Added %lld vectors in %lld ms", index->ntotal, endTime - startTime);

    delete[] allEmbeddings;

    // === Save the index ===
    LOGI("Saving index to %s", index_path.c_str());
    startTime = getCurrentMillTime();
    faiss::write_index(index, index_path.c_str());
    endTime = getCurrentMillTime();
    LOGI("Index saved in %lld ms", endTime - startTime);

    // Memory usage calculation
    float originalMemoryMB = (float)(index->ntotal * dimension * sizeof(float)) / (1024 * 1024);
    float compressedMemoryMB = (float)(index->ntotal * numSubvectors) / (1024 * 1024);
    float compressionRatio = originalMemoryMB / compressedMemoryMB;

    LOGI("Memory usage:");
    LOGI("  Original: %.2f MB", originalMemoryMB);
    LOGI("  Compressed: %.2f MB", compressedMemoryMB);
    LOGI("  Compression ratio: %.2f:1", compressionRatio);

    // Prepare result string
    result += "Compression Results:\n";
    result += "- Vectors: " + std::to_string(index->ntotal) + "\n";
    result += "- Dimension: " + std::to_string(dimension) + "\n";
    result += "- Original size: " + std::to_string(originalMemoryMB) + " MB\n";
    result += "- Compressed size: " + std::to_string(compressedMemoryMB) + " MB\n";
    result += "- Compression ratio: " + std::to_string(compressionRatio) + ":1\n";

    delete index;

    return env->NewStringUTF(result.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_io_datamachines_faiss_SentenceEmbeddingActivity_searchIndexNative(
        JNIEnv *env, jobject thiz, jfloatArray query_embedding, jint top_k, jstring index_path) {

    // Convert Java string to C++ string
    const char *cIndexPath = env->GetStringUTFChars(index_path, nullptr);
    std::string indexPath(cIndexPath);
    env->ReleaseStringUTFChars(index_path, cIndexPath);

    // Get the query embedding from Java
    jsize dimension = env->GetArrayLength(query_embedding);
    jfloat *queryEmbedding = env->GetFloatArrayElements(query_embedding, nullptr);

    // Load the Faiss index from the file
    LOGI("Loading index from %s", indexPath.c_str());
    faiss::Index *index = nullptr;
    try {
        index = faiss::read_index(indexPath.c_str());
    } catch (const std::exception &e) {
        LOGE("Failed to load index: %s", e.what());
        env->ReleaseFloatArrayElements(query_embedding, queryEmbedding, JNI_ABORT);
        return env->NewStringUTF("Error: Failed to load index.");
    }

    if (!index) {
        LOGE("Index is null after loading.");
        env->ReleaseFloatArrayElements(query_embedding, queryEmbedding, JNI_ABORT);
        return env->NewStringUTF("Error: Index is null.");
    }

    // Prepare for search
    LOGI("Performing search with top_k=%d", top_k);
    float *distances = new float[top_k];
    int64_t *indices = new int64_t[top_k];

    try {
        // Perform the search
        index->search(1, queryEmbedding, top_k, distances, indices);

        // Build the result string
        std::string resultJson;
        resultJson += "[";
        std::string result = "Search Results:\n";
        for (int i = 0; i < top_k; i++) {
            result += "Rank " + std::to_string(i + 1) + ": ";
            result += "ID=" + std::to_string(indices[i]) + ", ";
            result += "Distance=" + std::to_string(distances[i]) + "\n";
            resultJson += std::to_string(indices[i]);
            if (i < top_k - 1) resultJson += ",";
        }
        resultJson += "]";

        // Clean up and return results
        delete[] distances;
        delete[] indices;
        env->ReleaseFloatArrayElements(query_embedding, queryEmbedding, JNI_ABORT);
        delete index;

        return env->NewStringUTF(resultJson.c_str());

    } catch (const std::exception &e) {
        LOGE("Search failed: %s", e.what());
        delete[] distances;
        delete[] indices;
        env->ReleaseFloatArrayElements(query_embedding, queryEmbedding, JNI_ABORT);
        delete index;

        return env->NewStringUTF("Error: Search failed.");
    }
}



#define JNIREG_CLASS_BASE "io/datamachines/faiss/MainActivity"
static JNINativeMethod gMethods_Base[] = {
        {"stringFromJNI", "(ILjava/lang/String;)Ljava/lang/String;", (void *) stringFromJNI},
};

static JNINativeMethod gMethods_SentenceEmbedding[] = {
        {"compressEmbeddingsWithPQ",
         "([[FILjava/lang/String;)Ljava/lang/String;",
         (void *) Java_io_datamachines_faiss_SentenceEmbeddingActivity_compressEmbeddingsWithPQ},
        {"compressEmbeddingWithPQ",
         "([FLjava/lang/String;)Ljava/lang/String;",
         (void *) Java_io_datamachines_faiss_SentenceEmbeddingActivity_compressEmbeddingWithPQ},
        {"searchIndexNative",
                "([FILjava/lang/String;)Ljava/lang/String;",
                (void *) Java_io_datamachines_faiss_SentenceEmbeddingActivity_searchIndexNative}
};

static int registerNativeMethods(JNIEnv *env, const char *className,
                                 JNINativeMethod *gMethods, int numMethods) {
    jclass clazz;
    clazz = (*env).FindClass(className);
    if (clazz == nullptr) {
        return JNI_FALSE;
    }
    if ((*env).RegisterNatives(clazz, gMethods, numMethods) < 0) {
        return JNI_FALSE;
    }
    return JNI_TRUE;
}


static int registerNatives(JNIEnv *env) {
    if (!registerNativeMethods(env, JNIREG_CLASS_BASE, gMethods_Base,
                               sizeof(gMethods_Base) / sizeof(gMethods_Base[0]))) {
        return JNI_FALSE;
    }

    if (!registerNativeMethods(env, "io/datamachines/faiss/SentenceEmbeddingActivity",
                               gMethods_SentenceEmbedding,
                               sizeof(gMethods_SentenceEmbedding) / sizeof(gMethods_SentenceEmbedding[0]))) {
        return JNI_FALSE;
    }

    return JNI_TRUE;
}


JNIEXPORT jint

JNICALL JNI_OnLoad(JavaVM *vm, void *reserved) {
    LOGI("JNI_OnLoad");
    JNIEnv *env = nullptr;
    jint result = -1;

    if ((*vm).GetEnv((void **) &env, JNI_VERSION_1_4) != JNI_OK) {
        return -1;
    }
    assert(env != nullptr);

    if (!registerNatives(env)) { //注册
        return -1;
    }

    result = JNI_VERSION_1_4;
    return result;

}


JNIEXPORT void JNI_OnUnload(JavaVM *vm, void *reserved) {


}


