import random
import numpy as np

if __name__ == "__main__":
    print("Nope")
else:
    def get_batch(x, y, batch_size, end, start=0):
        random.seed(None)
        random_ids = random.sample(range(start, end), batch_size)
        return x[random_ids].transpose(1, 0), y[random_ids].transpose(1, 0)
        
    def translate_sentence(model, input):
        pass