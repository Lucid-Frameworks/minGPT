
# tabGPT

An adaption of Andrej Karpathy's minGPT for tabular data.

procedure:
- create column embedding for each column:
    - sentence embedding of column name/description (forward call of GPT2 with language-pre-trained weights and extract mean of last hidden states for each token in sequence)
    - for categorical values: for each row, same embedding procedure as for column name, then add the two embeddings
    - for numerical values: multiply the numerical values to each element of the column name embedding
- concatenation of the embeddings for the different column along sequence dimension
- run adjusted minGPT
    - instead of tokenization and learned embeddings, take the column embeddings as input
    - without positional encoding
    - instead of next-token prediction, use a classification or regression head (same as GPT2ForSequenceClassification in Hugging Face's transformers)


### License

MIT
