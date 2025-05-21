import spacy
import networkx as nx
import numpy as np
import tkinter as tk
from tkinter import scrolledtext
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
    
    def summarize_notes(self, complexity='medium'):
        """Uses Latent Semantic Analysis (LSA) to extract key topics from stored documents with adaptive complexity."""
        if not self.documents:
            return "No documents available."
        
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(self.documents)
        
        components = {'low': 1, 'medium': 2, 'high': 3}.get(complexity, 2)
        svd = TruncatedSVD(n_components=components)
        X_reduced = svd.fit_transform(X)
        
        top_terms = np.argsort(svd.components_[0])[-5:]
        keywords = [vectorizer.get_feature_names_out()[i] for i in top_terms]
        
        return f"Key Topics ({complexity} complexity): {', '.join(keywords)}"
    
    def retrieve_connections(self, query):
        """Retrieves related concepts from the knowledge graph."""
        if query in self.knowledge_graph:
            return list(self.knowledge_graph.neighbors(query))
        return "No related concepts found."
    
    def display_graph(self):
        """Prints nodes and edges of the knowledge graph."""
        return list(self.knowledge_graph.edges)

def run_gui():
    bot = TheoreticalNoteBot()
    
    def process_input():
        text = input_text.get("1.0", tk.END).strip()
        if text:
            bot.process_text(text)
            output_text.insert(tk.END, "Processed text successfully.\n")
    
    def summarize():
        complexity = complexity_var.get()
        summary = bot.summarize_notes(complexity)
        output_text.insert(tk.END, summary + "\n")
    
    def retrieve():
        query = query_entry.get().strip()
        results = bot.retrieve_connections(query)
        output_text.insert(tk.END, f"Connections for '{query}': {results}\n")
    
    root = tk.Tk()
    root.title("Theoretical Note Bot")
    
    tk.Label(root, text="Enter Text:").pack()
    input_text = scrolledtext.ScrolledText(root, height=5, width=50)
    input_text.pack()
    tk.Button(root, text="Process Text", command=process_input).pack()
    
    tk.Label(root, text="Select Complexity:").pack()
    complexity_var = tk.StringVar(value='medium')
    tk.OptionMenu(root, complexity_var, 'low', 'medium', 'high').pack()
    tk.Button(root, text="Summarize Notes", command=summarize).pack()
    
    tk.Label(root, text="Retrieve Connections:").pack()
    query_entry = tk.Entry(root)
    query_entry.pack()
    tk.Button(root, text="Retrieve", command=retrieve).pack()
    
    output_text = scrolledtext.ScrolledText(root, height=10, width=50)
    output_text.pack()
    
    root.mainloop()

if __name__ == "__main__":
    run_gui()
