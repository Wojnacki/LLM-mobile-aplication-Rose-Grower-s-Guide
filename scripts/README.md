flowchart LR
    A[knowledge/row<br/>poradnik_pielegnacji_roz_v_2.md]
    B[chunk_markdown.py<br/>chunking MD]
    C[fix_small_chunks.py<br/>scalanie małych chunków]
    D[validate_chunks.py<br/>walidacja jakości]
    E[build_faiss_index.py<br/>embedding + FAISS]
    F[FAISS index<br/>rose_index.faiss]
    G[test_rag_search.py<br/>manualne testy]
    H[rag_quality_eval.py<br/>automatyczna ewaluacja]

    A --> B --> C --> D --> E --> F
    F --> G
    F --> H
