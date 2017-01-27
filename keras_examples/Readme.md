# Keras Examples


This folder includes two sentiment analysis tests done with Keras. One example uses the imdb film comments dataset implemented in Keras, as [here](https://github.com/fchollet/keras/blob/master/examples/imdb_cnn_lstm.py) described and other example uses the Criticas Cine dataset (**in Spanish**).

#### How data is processed:
Both examples use the same apporach: words are replaced by their respective occurrences in the dataset, for example, if word "have" is the third most frequent word, then it is replaced by 3. By doing so, words are replaced by numbers that serve as input for the neural network.

**IMPORTANT:** in case of the dataset in Spanish (Criticas Cine Dataset), since the dataset is so small, the lemmatized text is used in order to reduce the vocabulary of the dataset.

#### Model architecture:
With Keras, it can be summarized as follows:
```
model = Sequential()
model.add(Embedding(max_features, embedding_size, input_length=maxlen))
model.add(Dropout(0.05))
model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
model.add(MaxPooling1D(pool_length=pool_length))
model.add(LSTM(lstm_output_size))
model.add(Dense(1))
model.add(Activation('sigmoid'))
  
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
```

Basically, the model architecture includes:
1. Embedding Layer: Takes words (in this case numbers!) from a vocabulary as input and embeds them as vectors into a lower dimensional space. [Here](http://sebastianruder.com/word-embeddings-1/) there is a detailed description of Embeddings.
2. Convolution Layer: Convolution operator for filtering neighborhoods of one-dimensional inputs.
3. MaxPooling: Max pooling operation for temporal data.
4. LSTM Layer: Long-Short Term Memory unit - Hochreiter 1997. This is an [excellent tutorial to describe how it works!](http://deeplearning.net/tutorial/lstm.html)
5. Output node with Activation Sigmoid: Returns 1 (positive sentiment) or 0 (negative sentiment)

Loss is calculated with ```binary_crossentropy``` optimized with ```adam``` and metric used is ```accuracy```.



#### Results:

**imdb dataset:**

|Training Size|Dropout|Epochs|Batch Size|Test Set Accuracy|
|-------------|-------|------|----------|-----------------|
|25000|0.25|2|30|0.86|

**Criticas Cine dataset:**

|Training Size|Dropout|Epochs|Batch Size|Test Set Accuracy|
|-------------|-------|------|----------|-----------------|
|3102|0.05|10|50|0.66|

#### Conclusions:

With imdb dataset significantly better results are achieved, but since there is a significant difference in the dataset sizes, this can not be assumed to the use of Spanish or English.



