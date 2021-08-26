# Naive-Bayes-Spam-Filtering
A python implementation of a Naive Bayes spam filter for sorting text messages.

# Setup
1. Create virtual environment (e.g. `python3 -m venv venv` and `source venv/bin/activate`)
2. Install necessary packages with `pip install -r requirements.txt`
3. Run filter with `python3 -m Bayes`

# Explanation
The python script reads in messages from SMSSpamCollection. This is the collection of labeled messages, spam or ham.
To start, all data is preprocessed:
 1. All punctuation is removed and the messages are converted into arrays of words
 2. All characters are lowercased
 3. All stop words are removed
 4. All words are stemmed using the Porter Stemmer Algorithm

Next, the labeled data is split into train and test data. 1/5 of the data set is used for training.
Finally, two Naive Bayes approaches are used to analyse the train data for the likely hood of the precense of various words indicating that the message is spam or ham. This data is then used on the test data to make classifications. The labels are then checked to determine accuracy. The first method is Bag of Words, and the second method is the more involved TF-IDF method. TF-IDF is slightly more accurate, but both hover around 95% accuracy. This result is significantly more accurate than guessing.

# Author
Sean Cavalieri
