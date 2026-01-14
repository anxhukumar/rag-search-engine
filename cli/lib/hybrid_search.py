import os
from lib import config
from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(config.INDEX_CACHE_FILE_PATH):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        bm25_score: list[tuple[int, int]] = self._bm25_search(query, limit * 500)
        sem_chunks: list[dict] = self.semantic_search.search_chunks(query, limit * 500)

        bm25_score_list = []
        for bm in bm25_score:
            bm25_score_list.append(bm[1])
        normalized_bm25_scores = normalize(bm25_score_list)

        sem_score_list = []
        for s in sem_chunks:
            sem_score_list.append(s["score"])
        normalized_sem_scores = normalize(sem_score_list)

        # Store docID -> {documents, normalized_keyword_score, normalized_semantic_score}
        combined_norm_scores = {}
        for i in range(len(bm25_score)):
            doc_id = bm25_score[i][0]
            norm_bm25_score = normalized_bm25_scores[i]
            combined_norm_scores[doc_id] = {"keyword_score": norm_bm25_score, "hybrid_score": 0.0, "doc": self.semantic_search.document_map[doc_id]}
        for i in range(len(sem_chunks)):
            doc_id = sem_chunks[i]["id"]
            norm_sem_score = normalized_sem_scores[i]
            if doc_id in combined_norm_scores:
                bm25_score = combined_norm_scores[doc_id]["keyword_score"]
                hybrid_score_val = hybrid_score(bm25_score, norm_sem_score, alpha)
                combined_norm_scores[doc_id]["semantic_score"] = norm_sem_score
                combined_norm_scores[doc_id]["hybrid_score"] = hybrid_score_val
            else:
                hybrid_score_val = hybrid_score(bm25_score, norm_sem_score, alpha)
                combined_norm_scores[doc_id] = {"keyword_score": 0.0, "hybrid_score": hybrid_score_val, "semantic_score": norm_sem_score, "doc": self.semantic_search.document_map[doc_id]}
        
        sorted_dict_list = dict(
            sorted(combined_norm_scores.items(), key=lambda x: x[1]["hybrid_score"], reverse=True)[:limit]
        )

        return sorted_dict_list

    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")
    
def normalize(scores: list[float]) -> list[float]:
    res = []
    if len(scores) == 0:
        return res

    min_score = min(scores)
    max_score = max(scores)

    if min_score == max_score:
        for _ in range(len(scores)):
            res.append(1.0)
        return res
    
    for s in scores:
        normalized_score = (s - min_score) / (max_score - min_score)
        res.append(normalized_score)
    
    return res

def hybrid_score(bm25_score, semantic_score, alpha=0.5):
    return alpha * bm25_score + (1 - alpha) * semantic_score