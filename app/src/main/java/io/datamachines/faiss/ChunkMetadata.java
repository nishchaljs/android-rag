package io.datamachines.faiss;

import android.net.Uri;

public class ChunkMetadata {
    private String pdfFilePath; // Path to the original PDF
    private int pageNumber;     // Page number in the PDF
    private int startOffset;    // Start offset of the chunk in characters
    private int endOffset;      // End offset of the chunk in characters

    public int getEndOffset() {
        return endOffset;
    }

    public int getStartOffset() {
        return startOffset;
    }

    public int getPageNumber() {
        return pageNumber;
    }

    public String getPdfFilePath() {
        return pdfFilePath;
    }

    public ChunkMetadata(String pdfFilePath, int pageNumber, int startOffset, int endOffset) {
        this.pdfFilePath = pdfFilePath;
        this.pageNumber = pageNumber;
        this.startOffset = startOffset;
        this.endOffset = endOffset;
    }

}

