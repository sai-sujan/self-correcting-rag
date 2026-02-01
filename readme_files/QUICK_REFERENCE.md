# Multi-Agent RAG System - Quick Reference Card

## 🎯 Key Dimensions at a Glance

### Vectors
| Component | Dimension | Memory | Type |
|-----------|-----------|--------|------|
| Dense Embedding | 768 | 3 KB | float32 |
| Sparse Embedding | 10K-50K | ~5 KB | sparse |

### Text Chunking
| Type | Size | Overlap | Tokens |
|------|------|---------|--------|
| Parent | 1,500 chars | 300 chars | ~375 |
| Child | 500 chars | 100 chars | ~125 |

### LLM Limits
| Parameter | Value |
|-----------|-------|
| Max Context | 128K tokens |
| Working Limit | 16K tokens |
| Max Output | 2,048 tokens |
| System Prompt | ~500 tokens |
| Retrieval Context | ~3,000 tokens |

### Retrieval
| Parameter | Value |
|-----------|-------|
| Top-K Results | 6 chunks |
| Dense Weight | 0.7 (70%) |
| Sparse Weight | 0.3 (30%) |
| Relevance Threshold | 7.0/10 |

### Performance Targets
| Metric | Target |
|--------|--------|
| P50 Latency | <4 sec |
| P90 Latency | <6 sec |
| Precision@6 | >0.7 |
| Faithfulness | >8.0/10 |

### Memory Management
| Parameter | Value |
|-----------|-------|
| Max Messages | 10 (5 user + 5 assistant) |
| Token Limit | 2,000 tokens |
| Compression Ratio | 7:1 |

## 🔧 Quick Commands

### Start Services
```bash
# Start Qdrant
docker run -p 6333:6333 qdrant/qdrant

# Start Ollama
ollama serve

# Pull models
ollama pull nomic-embed-text
ollama pull llama3.2
```

### Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run System
```bash
# Ingest documents
python src/ingest_documents.py --input data/pdfs/

# Start chatbot
python src/main.py
```

## 📊 Monitoring Checklist

- [ ] Qdrant running on port 6333
- [ ] Ollama running on port 11434
- [ ] Models downloaded (nomic-embed-text, llama3.2)
- [ ] Python environment activated
- [ ] LangSmith API key set (optional)
- [ ] Documents ingested

## 🐛 Common Issues

**Issue**: Slow inference
- **Fix**: Ensure GPU is being used, reduce batch size

**Issue**: Out of memory
- **Fix**: Reduce context window to 8K tokens

**Issue**: Poor retrieval quality
- **Fix**: Adjust dense/sparse weights, check chunk sizes

**Issue**: Connection refused (Qdrant)
- **Fix**: Check Docker container status

## 📈 Optimization Tips

1. **Cache embeddings** for frequently queried text
2. **Batch process** documents (32 at a time)
3. **Prune conversation** every 5 messages
4. **Monitor token usage** to stay under limits
5. **Use GPU** for 3-5x speed improvement

