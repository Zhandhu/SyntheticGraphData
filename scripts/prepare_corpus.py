"""Download a starter corpus of Wikipedia articles for development/testing."""

import json
import os
import sys
from src.utils.key_setup import ensure_keys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DOMAINS = {
    "physics": [
        "Quantum mechanics", "General relativity", "Standard Model",
        "Thermodynamics", "Electromagnetism", "Nuclear physics",
        "Particle physics", "Condensed matter physics",
        "Astrophysics", "String theory",
    ],
    "chemistry": [
        "Organic chemistry", "Periodic table", "Chemical bond",
        "Electrochemistry", "Polymer chemistry", "Catalysis",
        "Photochemistry", "Nuclear chemistry",
        "Biochemistry", "Green chemistry",
    ],
    "biology": [
        "DNA", "Evolution", "Cell biology", "Genetics",
        "Immunology", "Neuroscience", "Ecology",
        "Molecular biology", "Microbiology", "Photosynthesis",
    ],
    "mathematics": [
        "Calculus", "Linear algebra", "Number theory",
        "Topology", "Graph theory", "Probability theory",
        "Abstract algebra", "Differential equation",
        "Game theory", "Cryptography",
    ],
    "history": [
        "World War II", "French Revolution", "Roman Empire",
        "Industrial Revolution", "Cold War", "Renaissance",
        "Ancient Egypt", "American Civil War",
        "Silk Road", "Ottoman Empire",
    ],
    "geography": [
        "Plate tectonics", "Amazon rainforest", "Sahara",
        "Great Barrier Reef", "Himalayas", "Pacific Ocean",
        "Arctic", "Mediterranean Sea",
        "Ring of Fire", "Nile",
    ],
    "economics": [
        "Macroeconomics", "Supply and demand", "Monetary policy",
        "International trade", "Game theory", "Inflation",
        "Gross domestic product", "Stock market",
        "Federal Reserve", "European Central Bank",
    ],
    "computer_science": [
        "Machine learning", "Algorithm", "Computer network",
        "Operating system", "Database", "Artificial intelligence",
        "Cryptography", "Compiler",
        "Computer architecture", "Internet",
    ],
    "political_science": [
        "Democracy", "United Nations", "International law",
        "Political philosophy", "Federalism", "Human rights",
        "European Union", "NATO",
        "Sovereignty", "Geopolitics",
    ],
    "engineering": [
        "Civil engineering", "Electrical engineering",
        "Mechanical engineering", "Chemical engineering",
        "Aerospace engineering", "Nuclear engineering",
        "Biomedical engineering", "Software engineering",
        "Robotics", "Nanotechnology",
    ],
}


def main():
    try:
        import wikipedia
    except ImportError:
        print("Installing wikipedia package...")
        os.system(f"{sys.executable} -m pip install wikipedia")
        import wikipedia

    corpus_dir = "data/corpus/"
    os.makedirs(corpus_dir, exist_ok=True)

    metadata = []
    total = 0
    failed = 0

    for domain, topics in DOMAINS.items():
        print(f"\nDownloading {domain} articles...")
        for topic in topics:
            try:
                page = wikipedia.page(topic, auto_suggest=True)
                filename = f"{domain}_{topic.replace(' ', '_').lower()}.txt"
                filepath = os.path.join(corpus_dir, filename)

                with open(filepath, "w") as f:
                    f.write(page.content)

                metadata.append({
                    "title": page.title,
                    "url": page.url,
                    "domain": domain,
                    "filename": filename,
                    "length": len(page.content),
                })
                total += 1
                print(f"  ✓ {page.title}")

            except Exception as e:
                failed += 1
                print(f"  ✗ {topic}: {e}")

    # Save metadata
    meta_path = os.path.join(corpus_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone! Downloaded {total} articles, {failed} failed.")
    print(f"Saved to {corpus_dir}")
    print(f"Metadata at {meta_path}")


if __name__ == "__main__":
    ensure_keys()
    main()
