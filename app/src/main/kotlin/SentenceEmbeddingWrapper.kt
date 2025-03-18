package io.datamachines.faiss

import com.ml.shubham0204.sentence_embeddings.SentenceEmbedding
import kotlinx.coroutines.runBlocking

class SentenceEmbeddingWrapper {
    private val sentenceEmbedding = SentenceEmbedding()

    // Non-suspending init function that wraps the suspend function
    fun initSync(
        modelFilepath: String,
        tokenizerBytes: ByteArray,
        useTokenTypeIds: Boolean,
        outputTensorName: String,
        normalizeEmbedding: Boolean
    ): Int {
        return runBlocking {
            sentenceEmbedding.init(
                modelFilepath,
                tokenizerBytes,
                useTokenTypeIds,
                outputTensorName,
                normalizeEmbeddings = normalizeEmbedding
            )
        }
    }

    // Non-suspending encode function
    fun encodeSync(text: String): FloatArray {
        return runBlocking {
            sentenceEmbedding.encode(text)
        }
    }
}