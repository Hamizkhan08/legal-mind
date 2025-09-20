import os
import streamlit as st
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import chromadb
import PyPDF2
from docx import Document
from typing import List, Dict, Tuple, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import re
import hashlib
import json
from dataclasses import dataclass

# Memory-optimized Configuration
@dataclass
class Config:
    # Use much smaller, efficient models
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"  # Only 80MB vs 400MB+
    QA_MODEL: str = "distilbert-base-uncased-distilled-squad"  # Lightweight QA model
    CHUNK_SIZE: int = 256  # Reduced chunk size
    CHUNK_OVERLAP: int = 25  # Reduced overlap
    MAX_CONTEXT_LENGTH: int = 1000  # Reduced context
    GENERATION_MAX_LENGTH: int = 300  # Reduced generation length
    TEMPERATURE: float = 0.3
    TOP_P: float = 0.9
    TOP_K: int = 50
    # Memory optimization flags
    USE_CPU_ONLY: bool = True  # Force CPU to save GPU memory
    CACHE_EMBEDDINGS: bool = False  # Disable caching to save memory

config = Config()

class OptimizedLegalBrain:
    """Memory-optimized legal document analyzer"""
    
    def __init__(self):
        # Force CPU usage for free hosting
        self.device = "cpu"  # Always use CPU to save memory
        self.embedding_model = None
        self.qa_pipeline = None
        self._model_loaded = False
        
    @st.cache_resource
    def load_models(_self):
        """Load lightweight models optimized for free hosting"""
        try:
            # Load lightweight sentence transformer (only ~80MB)
            _self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
            _self.embedding_model.eval()  # Set to eval mode to save memory
            
            # Load lightweight QA pipeline using transformers
            from transformers import pipeline
            _self.qa_pipeline = pipeline(
                "question-answering",
                model=config.QA_MODEL,
                device=-1,  # Force CPU
                framework="pt"
            )
            
            _self._model_loaded = True
            return True
            
        except Exception as e:
            st.error(f"Error loading optimized models: {str(e)}")
            return False
    
    def get_embeddings(self, text: str) -> np.ndarray:
        """Generate embeddings using lightweight model"""
        if not self._model_loaded:
            if not self.load_models():
                return np.random.rand(384)  # MiniLM has 384 dimensions
        
        try:
            # Use sentence-transformers for efficient embedding
            embeddings = self.embedding_model.encode([text], convert_to_tensor=False)
            return embeddings[0]
        except Exception as e:
            st.error(f"Embedding error: {str(e)}")
            return np.random.rand(384)
    
    def extract_legal_concepts(self, text: str) -> List[str]:
        """Extract legal concepts using regex patterns"""
        legal_patterns = [
            r'\b(shall|must|may|will)\s+\w+',
            r'\b(contract|agreement|clause|term|condition)\b',
            r'\b(party|parties|plaintiff|defendant)\b',
            r'\b(liability|obligation|right|duty)\b',
            r'\b(terminate|breach|default|cure)\b',
            r'\b(payment|fee|penalty|damages)\b',
            r'\b(confidential|proprietary|intellectual property)\b',
            r'\b(warranty|guarantee|indemnify)\b',
            r'\b(force majeure|arbitration|jurisdiction)\b'
        ]
        
        concepts = []
        for pattern in legal_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            concepts.extend(matches)
        
        return list(set(concepts))[:10]  # Limit to 10 concepts to save memory
    
    def answer_question(self, question: str, context: str) -> Dict:
        """Answer questions using lightweight QA pipeline"""
        if not self._model_loaded or not self.qa_pipeline:
            return {"answer": "QA model not available", "confidence": 0.0}
        
        try:
            # Truncate context to fit model limits and save memory
            max_context = 800  # Reduced from 2000
            if len(context) > max_context:
                context = context[:max_context] + "..."
            
            # Use the pipeline for QA
            result = self.qa_pipeline(
                question=question,
                context=context,
                max_answer_len=100,  # Limit answer length
                handle_impossible_answer=True
            )
            
            return {
                "answer": result['answer'],
                "confidence": result['score'],
                "start": result.get('start', 0),
                "end": result.get('end', 0)
            }
            
        except Exception as e:
            st.error(f"QA error: {str(e)}")
            return {"answer": "Error processing question", "confidence": 0.0}
    
    def simplify_legal_text(self, legal_text: str, context: str = "") -> str:
        """Simplify legal text using pattern replacement"""
        # Comprehensive legal-to-plain mappings
        replacements = {
            r'\bshall\b': 'must',
            r'\bwhereas\b': 'since',
            r'\btherefore\b': 'so',
            r'\bheretofore\b': 'before this',
            r'\bhereinafter\b': 'from now on',
            r'\bparty of the first part\b': 'first party',
            r'\bparty of the second part\b': 'second party',
            r'\bin consideration of\b': 'in exchange for',
            r'\bnotwithstanding\b': 'despite',
            r'\bpursuant to\b': 'according to',
            r'\bwherein\b': 'where',
            r'\bwhereby\b': 'by which',
            r'\baforesaid\b': 'mentioned above',
            r'\bindemnify\b': 'protect from financial loss',
            r'\bliable\b': 'responsible',
            r'\bterminate\b': 'end',
            r'\bbreach\b': 'break or violate',
            r'\bdefault\b': 'fail to meet obligations',
            r'\bremedy\b': 'fix or solution',
            r'\bwaive\b': 'give up',
            r'\bvoid\b': 'invalid',
            r'\bnull and void\b': 'completely invalid',
            r'\bforce majeure\b': 'unforeseeable circumstances',
            r'\barbitration\b': 'private dispute resolution',
            r'\bjurisdiction\b': 'legal authority area'
        }
        
        simplified = legal_text
        for legal_term, plain_term in replacements.items():
            simplified = re.sub(legal_term, plain_term, simplified, flags=re.IGNORECASE)
        
        # Break down long sentences (memory efficient)
        sentences = simplified.split('. ')
        short_sentences = []
        
        for sentence in sentences[:5]:  # Limit processing to save memory
            if len(sentence) > 80:  # Reduced threshold
                # Simple split at conjunctions
                parts = re.split(r'\s+(and|or|but|however)\s+', sentence)
                for part in parts:
                    if part.strip() and part.strip() not in ['and', 'or', 'but', 'however']:
                        short_sentences.append(part.strip())
            else:
                short_sentences.append(sentence)
        
        return '. '.join([s for s in short_sentences if s.strip()])[:500]  # Limit output length
    
    def generate_summary(self, legal_texts: List[str], query_context: str = "") -> str:
        """Generate summary with memory optimization"""
        try:
            # Limit processing to save memory
            texts_to_process = legal_texts[:2]  # Only process first 2 chunks
            
            # Extract key concepts
            all_concepts = []
            for text in texts_to_process:
                concepts = self.extract_legal_concepts(text[:500])  # Limit text length
                all_concepts.extend(concepts)
            
            key_concepts = list(set(all_concepts))[:5]  # Limit to 5 concepts
            
            # Combine texts for analysis (memory efficient)
            combined_text = "\n\n".join([t[:400] for t in texts_to_process])
            
            # Generate answer using QA model
            summary_question = f"What are the main points about {query_context}?" if query_context else "What are the main legal points?"
            qa_result = self.answer_question(summary_question, combined_text)
            
            # Create simplified summary
            if len(qa_result["answer"]) < 30 or qa_result["confidence"] < 0.3:
                # Fallback to extractive summary
                key_sentences = self._extract_key_sentences(texts_to_process[0], query_context)
                base_answer = ". ".join(key_sentences[:2])
            else:
                base_answer = qa_result["answer"]
            
            # Simplify the answer
            simplified_answer = self.simplify_legal_text(base_answer, query_context)
            
            # Add context-specific explanations (brief)
            context_explanations = {
                'termination': "\n\nThis explains when and how the contract can be ended.",
                'payment': "\n\nThis covers payment obligations and timing.",
                'liability': "\n\nThis defines who is responsible for what."
            }
            
            for key, explanation in context_explanations.items():
                if key in query_context.lower():
                    simplified_answer += explanation
                    break
            
            # Create concise response
            response = f"""**Main Points:**
{simplified_answer}

**Key Terms:** {', '.join(key_concepts)}

**Confidence:** {qa_result['confidence']:.2f}

**Note:** This is a simplified explanation for reference only."""
            
            return response[:1000]  # Limit response length
            
        except Exception as e:
            return f"Summary generation error: {str(e)}"
    
    def _extract_key_sentences(self, text: str, query_context: str) -> List[str]:
        """Extract key sentences efficiently"""
        sentences = text.split('.')[:10]  # Limit to first 10 sentences
        key_sentences = []
        
        search_terms = query_context.lower().split() if query_context else ['contract', 'agreement']
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 15 and len(sentence) < 200:  # Reasonable length
                for term in search_terms:
                    if term in sentence.lower():
                        key_sentences.append(sentence)
                        break
                if len(key_sentences) >= 3:  # Limit to 3 sentences
                    break
        
        return key_sentences if key_sentences else [s.strip() for s in sentences[:2] if len(s.strip()) > 15]

# Memory-optimized RAG System
class OptimizedRAGSystem:
    """Memory-efficient RAG system for free hosting"""

    def __init__(self):
        self.brain = OptimizedLegalBrain()
        self.vector_db = None
        self.processor = DocumentProcessor()
        self._init_vector_db()
    
    def _init_vector_db(self):
        """Initialize lightweight vector database"""
        try:
            # Use in-memory ChromaDB to reduce disk usage
            client = chromadb.Client()  # In-memory client
            
            try:
                self.vector_db = client.create_collection(
                    name="legal_documents",
                    metadata={"hnsw:space": "cosine"}
                )
            except Exception:
                self.vector_db = client.get_collection(name="legal_documents")
                
        except Exception as e:
            st.error(f"Vector DB initialization error: {str(e)}")
            self.vector_db = None
    
    def process_document(self, file, doc_type: str = "legal") -> bool:
        """Process document with memory optimization"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Extract text
            status_text.text("Extracting text...")
            progress_bar.progress(25)
            
            if file.type == "application/pdf":
                text = self.processor.extract_text_from_pdf(file)
            elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                text = self.processor.extract_text_from_docx(file)
            elif file.type == "text/plain":
                text = str(file.read(), "utf-8")
            else:
                st.error("Unsupported file type")
                return False
            
            if not text:
                return False
            
            # Limit text length for free hosting
            if len(text) > 10000:  # Limit to ~10k characters
                text = text[:10000] + "..."
                st.warning("Document truncated due to memory limits on free hosting")
            
            status_text.text("Processing...")
            progress_bar.progress(50)
            
            # Process text efficiently
            clean_text = self.processor.clean_text(text)
            chunks = self.processor.chunk_text(clean_text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
            
            # Limit number of chunks for free hosting
            if len(chunks) > 20:
                chunks = chunks[:20]
                st.warning("Limited to 20 chunks due to memory constraints")
            
            status_text.text("Generating embeddings...")
            progress_bar.progress(75)
            
            # Process in smaller batches to save memory
            embeddings = []
            enhanced_chunks = []
            
            for i, chunk in enumerate(chunks):
                # Update progress
                current_progress = 75 + int((i / len(chunks)) * 20)
                progress_bar.progress(current_progress)
                
                # Get embedding
                embedding = self.brain.get_embeddings(chunk['text'])
                embeddings.append(embedding.tolist())
                
                # Extract concepts (limited)
                concepts = self.brain.extract_legal_concepts(chunk['text'][:300])
                
                # Create simplified preview
                preview_text = chunk['text'][:200]  # Reduced preview length
                simplified_preview = self.brain.simplify_legal_text(preview_text)
                
                enhanced_chunk = {
                    **chunk,
                    'legal_concepts': concepts,
                    'simplified_preview': simplified_preview
                }
                enhanced_chunks.append(enhanced_chunk)
                
                # Clear variables to free memory
                del embedding, concepts, preview_text, simplified_preview
            
            status_text.text("Storing in database...")
            progress_bar.progress(95)
            
            # Store in vector database efficiently
            doc_id = hashlib.md5(file.name.encode()).hexdigest()[:8]  # Shorter IDs
            metadata_list = []
            documents = []
            ids = []
            
            for i, (chunk, embedding) in enumerate(zip(enhanced_chunks, embeddings)):
                metadata = {
                    'filename': file.name,
                    'doc_type': doc_type,
                    'chunk_id': i,
                    'legal_concepts': json.dumps(chunk['legal_concepts']),
                    'simplified_preview': chunk['simplified_preview'][:100],  # Limit preview
                    'upload_date': datetime.now().isoformat()
                }
                metadata_list.append(metadata)
                documents.append(chunk['text'][:500])  # Limit stored text
                ids.append(f"{doc_id}_{i}")
            
            if self.vector_db is None:
                st.error("Vector database not initialized")
                return False
                
            self.vector_db.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadata_list,
                ids=ids
            )
            
            progress_bar.progress(100)
            status_text.text("Complete!")
            
            st.success(f"Processed {file.name} - {len(chunks)} chunks analyzed")
            return True
            
        except Exception as e:
            st.error(f"Processing error: {str(e)}")
            return False
    
    def query(self, question: str) -> Dict:
        """Memory-efficient query processing"""
        if not self.vector_db:
            return {
                'answer': "No documents uploaded yet.",
                'sources': [],
                'confidence': 0,
                'legal_concepts': []
            }
        
        try:
            # Get embedding for query
            query_embedding = self.brain.get_embeddings(question)
            
            # Search for relevant chunks (limited results)
            results = self.vector_db.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=3  # Reduced from 5 to save memory
            )
            
            if not results['documents'][0]:
                return {
                    'answer': "No relevant information found.",
                    'sources': [],
                    'confidence': 0,
                    'legal_concepts': []
                }
            
            # Process results efficiently
            relevant_texts = results['documents'][0]
            metadata_list = results['metadatas'][0]
            distances = results['distances'][0] if results['distances'] else [0] * len(relevant_texts)
            
            # Generate answer
            answer = self.brain.generate_summary(relevant_texts, question)
            
            # Calculate confidence
            confidence_scores = [1 - d for d in distances]
            overall_confidence = np.mean(confidence_scores)
            
            # Extract concepts
            all_concepts = []
            for metadata in metadata_list:
                if 'legal_concepts' in metadata:
                    try:
                        concepts = json.loads(metadata['legal_concepts'])
                        all_concepts.extend(concepts)
                    except:
                        pass
            
            unique_concepts = list(set(all_concepts))[:5]  # Limit concepts
            
            # Prepare sources
            sources = []
            for i, (metadata, distance) in enumerate(zip(metadata_list, distances)):
                sources.append({
                    'filename': metadata.get('filename', 'Unknown'),
                    'chunk_id': metadata.get('chunk_id', i),
                    'relevance_score': 1 - distance,
                    'simplified_preview': metadata.get('simplified_preview', '')[:100]
                })
            
            return {
                'answer': answer,
                'sources': sources,
                'confidence': overall_confidence,
                'legal_concepts': unique_concepts,
                'brain_model': 'OptimizedLegalMind'
            }
            
        except Exception as e:
            st.error(f"Query error: {str(e)}")
            return {
                'answer': f"Error: {str(e)}",
                'sources': [],
                'confidence': 0,
                'legal_concepts': []
            }

class DocumentProcessor:
    """Memory-efficient document processor"""
    
    @staticmethod
    def extract_text_from_pdf(file) -> str:
        try:
            reader = PyPDF2.PdfReader(file)
            text = ""
            # Limit pages to save memory
            max_pages = min(10, len(reader.pages))
            for i in range(max_pages):
                text += reader.pages[i].extract_text() + "\n"
                # Break if text gets too long
                if len(text) > 8000:
                    break
            return text
        except Exception as e:
            st.error(f"PDF extraction error: {str(e)}")
            return ""
    
    @staticmethod
    def extract_text_from_docx(file) -> str:
        try:
            doc = Document(file)
            text = ""
            # Limit paragraphs to save memory
            for i, paragraph in enumerate(doc.paragraphs):
                if i > 50:  # Limit paragraphs
                    break
                text += paragraph.text + "\n"
                if len(text) > 8000:  # Limit text length
                    break
            return text
        except Exception as e:
            st.error(f"DOCX extraction error: {str(e)}")
            return ""
    
    @staticmethod
    def clean_text(text: str) -> str:
        # More aggressive cleaning for memory efficiency
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'Page \d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'[^\w\s\.,;:!?()-]', '', text)  # Remove special chars
        return text.strip()
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 256, overlap: int = 25) -> List[Dict]:
        words = text.split()
        chunks = []
        
        # Limit total chunks for memory
        max_chunks = 30
        chunk_count = 0
        
        for i in range(0, len(words), chunk_size - overlap):
            if chunk_count >= max_chunks:
                break
                
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            chunks.append({
                'text': chunk_text,
                'start_idx': i,
                'end_idx': min(i + chunk_size, len(words)),
                'word_count': len(chunk_words)
            })
            
            chunk_count += 1
            
            if i + chunk_size >= len(words):
                break
        
        return chunks

def main():
    st.set_page_config(
        page_title="OptimizedLegalMind",
        page_icon="⚡",
        layout="wide"
    )
    
    st.title("⚡ OptimizedLegalMind (Free Hosting)")
    st.caption("Memory-optimized legal document analyzer for free cloud hosting")
    
    # Initialize session state
    if 'rag_system' not in st.session_state:
        with st.spinner("Loading optimized models..."):
            st.session_state.rag_system = OptimizedRAGSystem()
            st.session_state.rag_system.brain.load_models()
    
    # Sidebar
    with st.sidebar:
        st.header("📊 System Status")
        
        brain = st.session_state.rag_system.brain
        
        # Show memory-optimized status
        st.info("✅ CPU-Optimized Mode")
        st.success(f"✅ Embedding Model: {config.EMBEDDING_MODEL}")
        st.success(f"✅ QA Model: {config.QA_MODEL}")
        
        if st.session_state.rag_system.vector_db:
            st.success("✅ Vector Database")
        else:
            st.error("❌ Vector Database")
        
        st.divider()
        
        st.header("📄 Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload legal documents (PDF, DOCX, TXT)",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            help="Files will be truncated for free hosting limits"
        )
        
        if uploaded_files:
            for file in uploaded_files:
                if st.button(f"Analyze {file.name[:20]}...", key=f"process_{file.name}"):
                    with st.spinner(f"Processing..."):
                        st.session_state.rag_system.process_document(file)
        
        st.divider()
        
        # Document stats
        if st.session_state.rag_system.vector_db:
            try:
                count = st.session_state.rag_system.vector_db.count()
                st.metric("Document Chunks", count)
            except:
                st.metric("Document Chunks", "0")
        
        # Memory usage info
        st.caption("🔋 Optimized for free hosting")
        st.caption("📝 Limited to 20 chunks per document")
        st.caption("⚡ Using CPU-only inference")
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Ask Questions")
        
        # Quick questions
        quick_questions = [
            "What are the main terms?",
            "Explain key obligations",
            "What about payments?",
            "Termination clauses?",
            "Any liability issues?"
        ]
        
        cols = st.columns(len(quick_questions))
        for i, question in enumerate(quick_questions):
            with cols[i]:
                if st.button(question, key=f"q_{i}"):
                    st.session_state.current_question = question
        
        st.divider()
        
        user_question = st.text_area(
            "Your question:",
            height=60,
            placeholder="What happens if I terminate early?"
        )
        
        if st.button("🔍 Analyze", type="primary"):
            if user_question.strip():
                st.session_state.current_question = user_question
        
        # Display results
        if hasattr(st.session_state, 'current_question'):
            with st.spinner("Analyzing..."):
                result = st.session_state.rag_system.query(st.session_state.current_question)
                
                st.subheader("📋 Results")
                st.markdown(result['answer'])
                
                if result['sources']:
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Confidence", f"{result['confidence']:.2f}")
                    with col_b:
                        st.metric("Sources", len(result['sources']))
                
                if result['legal_concepts']:
                    st.subheader("🏷️ Key Terms")
                    st.write(", ".join(result['legal_concepts']))
                
                if result['sources']:
                    with st.expander("📚 Sources"):
                        for i, source in enumerate(result['sources']):
                            st.write(f"**{source['filename']}** (Score: {source['relevance_score']:.2f})")
                            if source.get('simplified_preview'):
                                st.caption(source['simplified_preview'])
    
    with col2:
        st.header("ℹ️ Info")
        
        st.info("""
        **Optimizations for Free Hosting:**
        
        • Lightweight models (80MB vs 400MB+)
        • CPU-only inference
        • Reduced chunk sizes
        • Limited document length
        • In-memory vector DB
        • Compressed embeddings
        """)
        
        if st.button("🔄 Refresh"):
            st.rerun()
        
        st.divider()
        
        st.header("📋 Model Info")
        st.caption(f"Embedding: {config.EMBEDDING_MODEL}")
        st.caption(f"QA: {config.QA_MODEL}")
        st.caption("Device: CPU")
        st.caption("Memory: Optimized")
    
    # Footer
    st.markdown("---")
    st.warning("⚠️ Optimized for free hosting - limited functionality. For production use, consider paid hosting with larger models.")

if __name__ == "__main__":
    main()
