#!/usr/bin/env python3
"""
RAG Pipeline Diagnostic Tool
Run this to verify all components are correctly configured
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

def check_env_variables():
    print("\n" + "="*60)
    print("1️⃣  ENVIRONMENT VARIABLES CHECK")
    print("="*60)
    
    required = {
        "GOOGLE_API_KEY": "Google Gemini API Key",
        "QDRANT_URL": "Qdrant Instance URL",
        "QDRANT_API_KEY": "Qdrant API Key",
        "JWT_SECRET": "JWT Secret (for auth)"
    }
    
    all_present = True
    for key, description in required.items():
        value = os.getenv(key)
        if value:
            masked = value[:5] + "*" * (len(value) - 10) + value[-5:] if len(value) > 10 else "*" * len(value)
            print(f"✅ {key}: {masked}")
        else:
            print(f"❌ {key}: MISSING - {description}")
            all_present = False
    
    return all_present

def check_embedding_model():
    print("\n" + "="*60)
    print("2️⃣  EMBEDDING MODEL CHECK")
    print("="*60)
    
    try:
        from Utils.utility import embedding_model, GeminiEmbeddings
        
        if isinstance(embedding_model, GeminiEmbeddings):
            print("✅ Using Custom GeminiEmbeddings wrapper (targeting text-embedding-004)")
        else:
            print(f"❌ Using {type(embedding_model).__name__} instead of GeminiEmbeddings")
            return False
        
        # Test embedding
        test_query = "test query"
        embedding = embedding_model.embed_query(test_query)
        dimension = len(embedding)
        
        if dimension == 768:
            print(f"✅ Embedding dimension: {dimension} (correct for text-embedding-004)")
            return True
        else:
            print(f"❌ Embedding dimension: {dimension} (should be 768 for text-embedding-004)")
            return False
            
    except Exception as e:
        print(f"❌ Error testing embedding model: {str(e)}")
        return False

def check_qdrant_connection():
    print("\n" + "="*60)
    print("3️⃣  QDRANT CONNECTION CHECK")
    print("="*60)
    
    try:
        from Utils.utility import _make_qdrant_client
        
        client = _make_qdrant_client()
        print(f"✅ Connected to Qdrant at {os.getenv('QDRANT_URL')}")
        
        # List collections
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        if collection_names:
            print(f"✅ Found {len(collection_names)} collection(s)")
            for name in collection_names:
                print(f"   - {name}")
        else:
            print("⚠️  No collections found. Need to run RAG/index.py first.")
        
        return True
    except Exception as e:
        print(f"❌ Could not connect to Qdrant: {str(e)}")
        print(f"   - Is QDRANT_URL correct? ({os.getenv('QDRANT_URL')})")
        print(f"   - Is QDRANT_API_KEY valid?")
        print(f"   - Is Qdrant instance running?")
        return False

def check_collection_content():
    print("\n" + "="*60)
    print("4️⃣  COLLECTION CONTENT CHECK")
    print("="*60)
    
    try:
        from Utils.utility import _make_qdrant_client, collection_name_for
        
        client = _make_qdrant_client()
        
        # Try discrete-mathematics collection
        coll = "discrete-mathematics"
        try:
            info = client.get_collection(coll)
            print(f"✅ Collection '{coll}' exists")
            
            # Check vector dimension
            vector_size = info.config.params.vectors.size
            if vector_size == 768:
                print(f"✅ Vector dimension: {vector_size} (correct)")
            else:
                print(f"❌ Vector dimension: {vector_size} (should be 768)")
                print(f"   This likely means vectors were indexed with different embedding model")
            
            # Check point count
            point_count = info.points_count
            if point_count > 0:
                print(f"✅ Collection has {point_count} vectors")
                return True
            else:
                print(f"❌ Collection is empty ({point_count} vectors)")
                print(f"   Run 'python RAG/index.py' to index documents")
                return False
                
        except Exception as e:
            print(f"❌ Collection '{coll}' not found: {str(e)}")
            print(f"   Run 'python RAG/index.py' to create and populate it")
            return False
            
    except Exception as e:
        print(f"❌ Error checking collection: {str(e)}")
        return False

def test_retrieval():
    print("\n" + "="*60)
    print("5️⃣  RETRIEVAL TEST")
    print("="*60)
    
    try:
        from Utils.utility import _make_qdrant_client, embedding_model
        from langchain_qdrant import QdrantVectorStore
        
        coll = "discrete-mathematics"
        client = _make_qdrant_client()
        vs = QdrantVectorStore(client=client, collection_name=coll, embedding=embedding_model)
        
        # Test query
        test_query = "What is discrete mathematics?"
        print(f"🔍 Test query: '{test_query}'")
        
        docs = vs.similarity_search(query=test_query, k=3)
        
        if docs:
            print(f"✅ Retrieved {len(docs)} documents")
            for i, doc in enumerate(docs):
                preview = doc.page_content[:60].replace('\n', ' ')
                print(f"   [{i+1}] {preview}...")
            return True
        else:
            print(f"❌ No documents retrieved")
            print(f"   - Check that 'discrete-mathematics' collection is not empty")
            print(f"   - Verify embedding model matches indexing model")
            print(f"   - Try re-running RAG/index.py")
            return False
            
    except Exception as e:
        print(f"❌ Error during retrieval: {str(e)}")
        return False

def main():
    print("\n" + "#"*60)
    print("# RAG PIPELINE DIAGNOSTIC TOOL")
    print("#"*60)
    
    results = []
    
    # 1. Check environment
    results.append(("Environment Variables", check_env_variables()))
    
    # 2. Check embedding model
    results.append(("Embedding Model", check_embedding_model()))
    
    # 3. Check Qdrant connection
    results.append(("Qdrant Connection", check_qdrant_connection()))
    
    # 4. Check collection content
    results.append(("Collection Content", check_collection_content()))
    
    # 5. Test retrieval
    results.append(("Retrieval Test", test_retrieval()))
    
    # Summary
    print("\n" + "="*60)
    print("📊 DIAGNOSTIC SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {check_name}")
    
    print(f"\nTotal: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 All checks passed! Your RAG pipeline is configured correctly.")
        return 0
    else:
        print("\n⚠️  Some checks failed. Review the errors above and follow the suggestions.")
        return 1

if __name__ == "__main__":
    sys.exit(main())