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

Example:
After 11 iterations. Diversity: 0.5. Generating with seed: "*modó su barba, y siguieron su camino, g*"
>modó su barba, y siguieron su camino, gentire, y para que no hay le dieren por la locura de la vista de la cual has asido a pastor de alguna pensamiento, y al diligerle, y si con la�perio su amor y con un escudero de mi mala caballería de todas cosas pasares el demanda, y el cual, en la lengua que tengo de así se le dejan su buena recio de la caballer�acho de pedir a la resi en que le amorecidad y en el primero de su casa de la mujer


### Conclusions:

* According to author comments:
At least 20 epochs are required before text starts sounding coherent.

* Training time is very slow on CPU
  * 8 hours for 3M characters. 
  * Minimum recommended 2.5 hours for 1M characters 




