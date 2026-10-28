# Research notes for the POS-tagger project

These are architecture references, not guarantees of performance on this assignment's split.

1. **dos Santos and Zadrozny (2014), Learning Character-level Representations for Part-of-Speech Tagging.** PMLR/ICML. Motivates combining trainable character-level spelling/morphology representations with learned word representations. The paper reports 97.32% English POS accuracy on its Penn Treebank WSJ setup; compare numbers carefully because tokenization, data split, preprocessing, and evaluation settings matter. https://proceedings.mlr.press/v32/santos14.html

2. **Ma and Hovy (2016), End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF.** ACL. Motivates combining a character CNN, BiLSTM contextual encoder, and CRF decoder. The paper reports 97.55% POS accuracy in its setting. The CRF decoder is an experimental option in this project, not assumed to be better in advance. https://aclanthology.org/P16-1101/

3. **Kemos, Adel, and Schütze (2019), Neural Semi-Markov Conditional Random Fields for Robust Character-Based Part-of-Speech Tagging.** NAACL-HLT. Character-level semi-Markov tagging is aimed at noisy token boundaries; our PA2 input is already tokenized, so this is useful context for robustness rather than a directly comparable model. https://aclanthology.org/N19-1280/

4. **Marcus, Santorini, and Marcinkiewicz (1993), Building a Large Annotated Corpus of English: The Penn Treebank.** Computational Linguistics. Describes the Penn Treebank corpus underlying the tagset/data conventions. https://aclanthology.org/J93-2004/

## Research questions tested by this code

- Are character features useful beyond a word embedding, particularly for exact-surface OOV tokens?
- Does a compact token-shape representation help capitalization/numeral/punctuation patterns?
- Does a character LSTM outperform a multi-width character CNN under the same input-feature budget?
- Does preserving case in lexical IDs help beyond an orthographic shape feature?
- Does a configuration close to the supplied peer report (80-D word + 20-D char, 192 hidden units, 16-D character embeddings) reproduce its reported behavior?
- Does right context (BiLSTM) help relative to a forward-only LSTM?
- Does sequence-level CRF decoding help beyond per-token softmax?
- How does word/character/shape dimension allocation affect validation accuracy and OOV accuracy?
- How sensitive are results to hidden size/depth, dropout, word dropout, learning rate, batch size, and label smoothing?

The experiment runner reports measured results from the project's own fixed validation split. Research-paper benchmark numbers must not be treated as a score on the supplied assignment development or blind test sets.
