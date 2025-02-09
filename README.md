# ChatBot for Students

## Introduction
ChatBot for Students is a system that helps students search for information from documents such as student handbooks, PDFs, and websites. The system utilizes **Retrieval-Augmented Generation (RAG)** to provide accurate answers based on available data.


## Concept Explanation
### RAG
- **Retrieval-Augmented Generation (RAG)** is a technique that enhances the ability of language model generation to combine with external knowledge.
- This method works by retrieving relevant information from the document (knowledge) store and using them for the answer generation process based on LLMs.

### TF-IDF (Term Frequency - Inverse Document Frequency)
- Measures the importance of a word in a document relative to a collection of documents (corpus).
- Helps identify significant terms by balancing their frequency in a document against their rarity across the corpus.

### Embedding Model
- Converts text into numerical vector representations in a multi-dimensional space.
- Enables semantic understanding and similarity-based search.
- Improves the accuracy of retrieving relevant documents.
## Pipeline
![image](https://github.com/user-attachments/assets/11d28fbf-82c2-409a-aca8-834c3887816b)

The processing pipeline consists of the following steps:
1. **Collect documents from various sources** (PDFs, URLs, student handbooks, etc.).
2. **Split documents into smaller chunks**.
3. **Generate embeddings(all-MiniLM-L6-v2) and TF-IDF representations** for the text chunks.
4. **Store data in a Vector Database and TF-IDF Index**.
5. **Receive user queries** and retrieve relevant information.
6. **Fetch the top k most relevant text chunks** from the database.
7. **Send the retrieved chunks to a Large Language Model (Gemini)** for response generation.
8. **Provide an accurate answer to the user** based on retrieved information.

Technologies used: **LangChain** - Manages the chatbot pipeline.

## Demo

<p align="center">
  <img src="https://github.com/user-attachments/assets/73a9f5b6-9b4b-4d5a-a489-d4e1fa9ccf64" width="45%" />
  <img src="https://github.com/user-attachments/assets/33408377-7802-4fff-a680-05b3107a04ae" width="45%" />
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/571ffd98-c163-489d-a6f2-337e7e4b6b5b" width="45%" />
</p>

