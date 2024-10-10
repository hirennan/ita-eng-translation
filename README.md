This project implements a machine translator for converting text from Italian to English. It is built using a sequence-to-sequence model with an encoder-decoder architecture. Sentences < 10 words were utilised to train the model faster.

## Results:
The trained model was tested on a split test dataset, and BLEU scores were calculated for each translated sentence to measure translation quality. The results indicate that the model performs reasonably well on ~38% of the sentences.

![image](https://github.com/user-attachments/assets/e8650c4e-ee3e-41af-bc77-b79a9033e646)

## Usage:
Use the command line to pass either "train", "test", "eval" to perform the respective tasks.

## Future work:
* Train the model on sentence lenghts > 10
* Improve translation quality using attention mechanism.
