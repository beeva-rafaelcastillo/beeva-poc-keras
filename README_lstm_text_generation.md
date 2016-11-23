# Keras example for text generation
Following example https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py with text from [El Quijote](http://www.gutenberg.org/cache/epub/2000/pg2000.txt)

### Step 1:
Truncate text corpus to 10K lines. Corpus stats:
* 102492 words
* 2198927 characters

### Step 2:

Parameters: 
* 40 characters/sequence
* 3 characters/step. So 732963 steps (sequences)
```
python lstm_text_generation.py
```

### Results:

Time per iteration (Intel Core i5): 1400s aprox.


### Conclusions:

* According to author comments:
At least 20 epochs are required before text starts sounding coherent.

* Training time is very slow on CPU
  * 8 hours for 3M characters. 
  * Minimum recommended 2.5 hours for 1M characters 




