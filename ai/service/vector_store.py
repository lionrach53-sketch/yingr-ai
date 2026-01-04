import faiss
import os
import pickle
import numpy as np

class VectorStore:
    def __init__(self, dim=384, path="data/faiss"):
        self.path = path
        self.index_path = os.path.join(path, "index.faiss")
        self.meta_path = os.path.join(path, "meta.pkl")
        os.makedirs(path, exist_ok=True)

        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, "rb") as f:
                self.meta = pickle.load(f)
        else:
            self.index = faiss.IndexFlatL2(dim)
            self.meta = []

    def add(self, vectors, metadata):
        vectors = np.array(vectors).astype("float32")
        self.index.add(vectors)
        self.meta.extend(metadata)
        self._save()

    def search(self, vector, k=5, return_scores=False):
        vector = np.array(vector).astype("float32")
        D, I = self.index.search(vector, k)
        # FAISS peut retourner -1 si pas assez d'éléments, ou des indices
        # qui dépassent la longueur de self.meta si l'index et les métadonnées
        # sont désynchronisés. On filtre pour éviter les crashs.
        results = []
        distances = []
        for dist, idx in zip(D[0].tolist(), I[0].tolist()):
            if idx is None or idx < 0:
                continue
            if idx >= len(self.meta):
                continue
            results.append(self.meta[idx])
            distances.append(dist)
        if return_scores:
            return results, distances  # distances L2 filtrées
        return results

    def _save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.meta, f)
    
    def get_all_metadata(self):
        """Retourne toutes les métadonnées avec index"""
        return [(i, meta) for i, meta in enumerate(self.meta)]
    
    def get_stats(self):
        """Statistiques sur l'index FAISS"""
        total_docs = self.index.ntotal
        languages = {}
        sources = {}
        
        for meta in self.meta:
            # Extraire langue depuis source (ex: admin-json-Histoire-fr)
            source = meta.get('source', '')
            if '-fr' in source:
                languages['fr'] = languages.get('fr', 0) + 1
            elif '-mo' in source:
                languages['mo'] = languages.get('mo', 0) + 1
            elif '-di' in source:
                languages['di'] = languages.get('di', 0) + 1
            else:
                languages['unknown'] = languages.get('unknown', 0) + 1
            
            # Compter par source
            sources[source] = sources.get(source, 0) + 1
        
        return {
            'total_documents': total_docs,
            'languages': languages,
            'sources': sources,
            'index_size_mb': os.path.getsize(self.index_path) / (1024*1024) if os.path.exists(self.index_path) else 0
        }
    
    def delete_by_indices(self, indices_to_delete):
        """Supprime des documents par leurs indices"""
        if not indices_to_delete:
            return

        # Normaliser les indices à supprimer
        # On se base sur la taille réelle de l'index FAISS pour éviter
        # les erreurs lorsque meta et index sont désynchronisés.
        ntotal = self.index.ntotal
        if ntotal == 0:
            # Rien à supprimer côté index, on vide simplement les métadonnées
            self.meta = []
            self._save()
            return

        indices_to_delete_set = {i for i in indices_to_delete if 0 <= i < ntotal}

        # FAISS ne supporte pas la suppression directe
        # Il faut recréer l'index sans les documents supprimés
        indices_to_keep = [i for i in range(ntotal) if i not in indices_to_delete_set]

        if not indices_to_keep:
            # Tous les docs supprimés, réinitialiser
            dim = self.index.d
            self.index = faiss.IndexFlatL2(dim)
            self.meta = []
        else:
            # Récupérer les vecteurs à garder
            vectors_to_keep = []
            meta_to_keep = []

            for i in indices_to_keep:
                # Récupérer le vecteur depuis l'index en toute sécurité
                vector = faiss.vector_to_array(self.index.reconstruct(i))
                vectors_to_keep.append(vector)
                # Protéger l'accès aux métadonnées si meta est plus court
                if i < len(self.meta):
                    meta_to_keep.append(self.meta[i])
                else:
                    meta_to_keep.append({"source": "unknown", "text": ""})

            # Créer nouvel index
            dim = self.index.d
            self.index = faiss.IndexFlatL2(dim)
            self.meta = []

            # Ré-ajouter les vecteurs conservés
            if vectors_to_keep:
                vectors_array = np.array(vectors_to_keep).astype('float32')
                self.index.add(vectors_array)
                self.meta = meta_to_keep

        self._save()
    
    def delete_by_source(self, source_pattern):
        """Supprime tous les documents d'une source spécifique"""
        indices_to_delete = []
        for i, meta in enumerate(self.meta):
            if source_pattern in meta.get('source', ''):
                indices_to_delete.append(i)
        
        self.delete_by_indices(indices_to_delete)
        return len(indices_to_delete)
    
    def delete_by_language(self, language):
        """Supprime tous les documents d'une langue spécifique (fr, mo, di)"""
        pattern = f'-{language}'
        return self.delete_by_source(pattern)
    
    def clear_all(self):
        """Vide complètement l'index RAG"""
        dim = self.index.d
        self.index = faiss.IndexFlatL2(dim)
        self.meta = []
        self._save()
