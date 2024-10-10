from seq2seq_model import get_data
import matplotlib.pyplot as plt
from collections import Counter

filtered_inp_lang, filtered_out_lang, inp_vocab, out_vocab = \
    get_data()

#plot sentence distributions

inp_lengths = [len(s.split()) for s in filtered_inp_lang]
out_lengths = [len(s.split()) for s in filtered_out_lang]


plt.figure(figsize=(10, 6))
plt.hist(inp_lengths, bins=20, alpha=0.5, label='English Sentence Lengths')
plt.hist(out_lengths, bins=20, alpha=0.5, label='Italian Sentence Lengths')
plt.xlabel('Sentence Length')
plt.ylabel('Number of Sentences')
plt.legend()
plt.title('Sentence Length Distribution')
plt.show()

#plot most frequent words

inp_word_counter = Counter(_w for _s in filtered_inp_lang \
                           for _w in _s.split())
out_word_counter = Counter(_w for _s in filtered_out_lang \
                           for _w in _s.split())
    
most_freq_inp_words, most_freq_out_words = \
    inp_word_counter.most_common(10), out_word_counter.most_common(10)

words, counts = zip(*most_freq_inp_words)
plt.figure(figsize=(10, 6))
plt.bar(words, counts)
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Top 10 Most Common Words')
plt.show()

words, counts = zip(*most_freq_out_words)
plt.figure(figsize=(10, 6))
plt.bar(words, counts)
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Top 10 Most Common Words')
plt.show()

#compare sentence lengths

plt.figure(figsize=(10, 6))
# plt.scatter(inp_lengths, out_lengths, alpha=0.2)
plt.hexbin(inp_lengths, out_lengths, gridsize=30, \
           cmap='Blues', mincnt=1)
plt.colorbar(label='Frequency')
plt.xlabel("English sentence lengths")
plt.ylabel("Italian sentence lengths")
plt.title("Eng vs Ita sentence lengths")
plt.show()

