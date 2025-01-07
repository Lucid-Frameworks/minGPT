import torch as th
from mingpt.bpe import BPETokenizer

# Input with only goal
# input = "Cristiano Ronaldo and Lionel Messi have both left an indelible mark on football through their incredible careers. Ronaldo has played for top clubs like Manchester United, Real Madrid, and Juventus, and currently plays for Al-Nassr. He has won numerous titles, including five Champions League titles and five Ballon d'Or awards. Messi, on the other hand, has spent the majority of his career at FC Barcelona before moving to Paris Saint-Germain. He has also achieved great success, with four Champions League titles and an impressive seven Ballon d'Or awards. As of now, Ronaldo has scored over 813 career goals, while Messi has netted around 798, showcasing the extraordinary talent and dedication of both players."
# Input with goal contributions
input = " Cristiano Ronaldo and Lionel Messi have both left an indelible mark on football through their incredible careers. Ronaldo has played for top clubs like Manchester United, Real Madrid, and Juventus, and currently plays for Al-Nassr. He has won numerous titles, including five Champions League titles and five Ballon d'Or awards. Messi, on the other hand, has spent the majority of his career at FC Barcelona before moving to Paris Saint-Germain. He has also achieved great success, with four Champions League titles and an impressive seven Ballon d'Or awards. As of now, Cristiano Ronaldo has a total goal contribution (goals + assists) of 1027, while Messi's goal contribution stands at 1092, showcasing the extraordinary talent and dedication of both players."
print("Input:", input)
bpe = BPETokenizer()
# bpe() gets a string and returns a 2D batch tensor 
# of indices with shape (1, input_length)
tokens = bpe(input)[0]
print("Tokenized input:", tokens)
input_length = tokens.shape[-1]
print("Number of input tokens:", input_length)
# bpe.decode gets a 1D tensor (list of indices) and returns a string
print("Detokenized input from indices:", bpe.decode(tokens))  
tokens_str = [bpe.decode(th.tensor([token])) for token in tokens]
print("Detokenized input as strings: " + '/'.join(tokens_str))