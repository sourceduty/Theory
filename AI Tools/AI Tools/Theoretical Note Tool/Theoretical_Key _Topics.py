import spacy
import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

class TheoreticalNoteBot:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.knowledge_graph = nx.DiGraph()
        self.documents = []
    
    def process_text(self, text):
        """Extracts key concepts from text and adds them to the knowledge graph."""
        doc = self.nlp(text)
        entities = set()
        
        for ent in doc.ents:
            self.knowledge_graph.add_node(ent.text, label=ent.label_)
            entities.add(ent.text)
        
        for i, ent1 in enumerate(entities):
            for j, ent2 in enumerate(entities):
                if i != j:
                    self.knowledge_graph.add_edge(ent1, ent2)
        
        self.documents.append(text)
    
    def summarize_notes(self):
        """Uses Latent Semantic Analysis (LSA) to extract key topics from stored documents."""
        if not self.documents:
            return "No documents available."
        
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(self.documents)
        
        svd = TruncatedSVD(n_components=2)
        X_reduced = svd.fit_transform(X)
        
        top_terms = np.argsort(svd.components_[0])[-5:]
        keywords = [vectorizer.get_feature_names_out()[i] for i in top_terms]
        
        return f"Key Topics: {', '.join(keywords)}"
    
    def display_graph(self):
        """Prints nodes and edges of the knowledge graph."""
        return list(self.knowledge_graph.edges)

# Example Usage
bot = TheoreticalNoteBot()
bot.process_text("Quantum mechanics is a fundamental theory in physics that describes nature at small scales.")
bot.process_text("Einstein's theory of relativity revolutionized physics and introduced spacetime curvature.")
print(bot.summarize_notes())
print(bot.display_graph())
