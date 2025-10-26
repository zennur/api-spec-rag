import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import re
from dataclasses import dataclass

@dataclass
class Document:
    """Represents a single document with metadata"""
    content: str
    file_path: str
    title: str
    section: Optional[str] = None
    
class APIDocsVectorStore:
    """
    A class for collecting markdown API documentation files and creating
    a vector store with embeddings for semantic search.
    """
    
    def __init__(
        self, 
        model_name: str = 'all-MiniLM-L6-v2',
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ):
        """
        Initialize the vector store.
        
        Args:
            model_name: Name of the sentence transformer model to use
            chunk_size: Maximum size of text chunks in characters
            chunk_overlap: Number of overlapping characters between chunks
        """
        self.model = SentenceTransformer(model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.documents: List[Document] = []
        self.chunks: List[str] = []
        self.chunk_metadata: List[Dict] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index: Optional[faiss.Index] = None
        
    def collect_markdown_files(self, directory: str) -> List[str]:
        """
        Recursively collect all markdown files from a directory.
        
        Args:
            directory: Root directory to search for .yaml files
            
        Returns:
            List of file paths to markdown files
        """
        yaml_files = []
        path = Path(directory)
        print(path)
        
        for file_path in path.rglob('*.yaml'):
            if file_path.is_file():
                yaml_files.append(str(file_path))
                
        print(f"Found {len(yaml_files)} yaml files")
        return yaml_files
    
    def parse_markdown(self, file_path: str) -> Document:
        """
        Parse a markdown file and extract content with metadata.
        
        Args:
            file_path: Path to the markdown file
            
        Returns:
            Document object with parsed content
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract title (first H1 heading)
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        title = title_match.group(1) if title_match else Path(file_path).stem
        
        return Document(
            content=content,
            file_path=file_path,
            title=title
        )
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + self.chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < text_length:
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > self.chunk_size // 2:
                    chunk = chunk[:break_point + 1]
                    end = start + break_point + 1
            
            chunks.append(chunk.strip())
            start = end - self.chunk_overlap
            
        return chunks
    
    def load_documents(self, directory: str):
        """
        Load and process all markdown documents from a directory.
        
        Args:
            directory: Root directory containing markdown files
        """
        yaml_files = self.collect_markdown_files(directory)
        
        for file_path in yaml_files:
            print(file_path)
            doc = self.parse_markdown(file_path)
            self.documents.append(doc)
            
            # Chunk the document
            doc_chunks = self.chunk_text(doc.content)
            
            for i, chunk in enumerate(doc_chunks):
                self.chunks.append(chunk)
                self.chunk_metadata.append({
                    'file_path': doc.file_path,
                    'title': doc.title,
                    'chunk_index': i,
                    'total_chunks': len(doc_chunks)
                })
        
        print(f"Loaded {len(self.documents)} documents")
        print(f"Created {len(self.chunks)} chunks")
    
    def create_embeddings(self):
        """
        Generate embeddings for all text chunks using the transformer model.
        """
        if not self.chunks:
            raise ValueError("No chunks to embed. Load documents first.")
        
        print("Generating embeddings...")
        self.embeddings = self.model.encode(
            self.chunks,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        print(f"Generated embeddings with shape: {self.embeddings.shape}")
    
    def build_index(self, index_type: str = 'flat'):
        """
        Build FAISS index for efficient similarity search.
        
        Args:
            index_type: Type of FAISS index ('flat' or 'ivf')
        """
        if self.embeddings is None:
            raise ValueError("No embeddings found. Create embeddings first.")
        
        dimension = self.embeddings.shape[1]
        
        if index_type == 'flat':
            # Simple flat index for exact search
            self.index = faiss.IndexFlatL2(dimension)
        elif index_type == 'ivf':
            # IVF index for faster approximate search
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, 100)
            self.index.train(self.embeddings)
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        self.index.add(self.embeddings)
        print(f"Built {index_type} index with {self.index.ntotal} vectors")
    
    def search(
        self, 
        query: str, 
        top_k: int = 5
    ) -> List[Tuple[str, Dict, float]]:
        """
        Search for similar documents using semantic similarity.
        
        Args:
            query: Search query text
            top_k: Number of top results to return
            
        Returns:
            List of tuples (chunk_text, metadata, similarity_score)
        """
        if self.index is None:
            raise ValueError("No index found. Build index first.")
        
        # Encode query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        
        # Search
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            results.append((
                self.chunks[idx],
                self.chunk_metadata[idx],
                float(dist)
            ))
        
        return results
    
    def save(self, save_path: str):
        """
        Save the vector store to disk.
        
        Args:
            save_path: Directory path to save the vector store
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        if self.index is not None:
            faiss.write_index(self.index, str(save_path / 'faiss.index'))
        
        # Save metadata and chunks
        with open(save_path / 'metadata.pkl', 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'chunk_metadata': self.chunk_metadata,
                'documents': self.documents,
                'embeddings': self.embeddings
            }, f)
        
        print(f"Saved vector store to {save_path}")
    
    def load(self, load_path: str):
        """
        Load a previously saved vector store.
        
        Args:
            load_path: Directory path containing the saved vector store
        """
        load_path = Path(load_path)
        
        # Load FAISS index
        index_path = load_path / 'faiss.index'
        if index_path.exists():
            self.index = faiss.read_index(str(index_path))
        
        # Load metadata and chunks
        with open(load_path / 'metadata.pkl', 'rb') as f:
            data = pickle.load(f)
            self.chunks = data['chunks']
            self.chunk_metadata = data['chunk_metadata']
            self.documents = data['documents']
            self.embeddings = data['embeddings']
        
        print(f"Loaded vector store from {load_path}")


# Example usage
# if __name__ == "__main__":
#     # Initialize the vector store
#     vectorizer = APIDocsVectorStore(
#         model_name='all-MiniLM-L6-v2',
#         chunk_size=512,
#         chunk_overlap=50
#     )
    
#     # Load documents from directory
#     vectorizer.load_documents('./api_docs')
    
#     # Create embeddings
#     vectorizer.create_embeddings()
    
#     # Build search index
#     vectorizer.build_index(index_type='flat')
    
#     # Save for later use
#     vectorizer.save('./vector_store')
    
#     # Search example
#     results = vectorizer.search("How do I authenticate API requests?", top_k=3)
    
#     for i, (chunk, metadata, score) in enumerate(results, 1):
#         print(f"\n--- Result {i} (Score: {score:.4f}) ---")
#         print(f"File: {metadata['file_path']}")
#         print(f"Title: {metadata['title']}")
#         print(f"Chunk: {metadata['chunk_index'] + 1}/{metadata['total_chunks']}")
#         print(f"Content: {chunk[:200]}...")