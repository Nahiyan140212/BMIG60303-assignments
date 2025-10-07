## Part 1
### Top Terms for Selected Topics:

**Topic 0:** in, 19, covid, to, is, with, disease, for, coronavirus, as
**Topic 5:** to, patients, in, with, for, care, is, patient, be, treatment
**Topic 7:** to, in, 19, covid, was, were, health, with, during, pandemic

### Analysis:

Common words like "in", "to", "with", "for" appear as top terms in nearly every topic. Topics are highly redundant with similar word distributions. Topic 5 hints at patient care/treatment, Topic 7 shows pandemic/health themes. Without stopword removal and better preprocessing, the topics are not coherent or interpretable.


## Part 2 (Experiment Analysis)
- Even 4% non-English content creates one completely useless topic. Language filtering is ESSENTIAL for interpretable topic modeling.
- Stopword removal is CRITICAL. Without it, topics capture only  grammar rather than meaning.

- Punctuation removal is MANDATORY. Without it, formatting characters
dominate topics and prevent meaningful analysis.

- Too few topics (5): Themes too broad, important distinctions lost
- Too many topics (15+): Redundancy increases, similar themes split artificially
- Optimal range: 10-12 topics for this corpus

## Part 3: PyLDAvis Visualization

- Topics are well-distributed in 2D space with minimal overlap. Clinical topics cluster together. Virology topics form a distinct group. Social science topics are separated from medical topics

- Relatively balanced distribution (ranging from ~6% to 10% of tokens)

- Topic 9 (Epidemiology & Modeling) represents 10.3% of tokens, showing the prevalence of epidemiological research in the corpus
