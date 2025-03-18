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

#include "log.h"
using namespace faiss;

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


#define JNIREG_CLASS_BASE "io/datamachines/faiss/MainActivity"
static JNINativeMethod gMethods_Base[] = {
        {"stringFromJNI", "(ILjava/lang/String;)Ljava/lang/String;", (void *) stringFromJNI},
};

static JNINativeMethod gMethods_SentenceEmbedding[] = {
        {"compressEmbeddingWithPQ", "([FLjava/lang/String;)Ljava/lang/String;",
         (void *) Java_io_datamachines_faiss_SentenceEmbeddingActivity_compressEmbeddingWithPQ},
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


