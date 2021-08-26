import numpy as np
import os
import nltk
import math
import random
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split

if __name__ == '__main__':

    # ---------- LAB PART 1 ----------

    # Ensure stopwords nltk data is accessable
    nltk.data.path.append(os.path.join('.','nltk_data'))
    # Open input and output files
    input = open("SMSSpamCollection","r")
    # input = open("spamTests.txt","r")
    proccessed = []
    # Split input file by line and loop for each
    lines = input.read().splitlines()
    for line in lines:
        # Remove punctuation and split into array of words (Note: Contractions get split in two, but the second parts are removed by stopwords step)
        tokenizer = nltk.RegexpTokenizer(r"\w+")
        wordsArr = tokenizer.tokenize(line)
        # Make all words lower case
        lowered = list(map(str.lower, wordsArr))
        # Remove stop words
        stop_words = nltk.corpus.stopwords.words("english")
        noStopWords = [w for w in lowered if not w in stop_words]
        # Stem all words
        ps = PorterStemmer()
        stemmed = list(map(ps.stem, noStopWords))
        # Recreate string and write to output file
        result = ' '.join(stemmed)
        proccessed.append(result)

    # ---------- LAB PART 2 ----------

    # Make numpy array from processed data
    linesNP = np.asarray(proccessed)
    # Split into train and test data
    train, test = train_test_split( linesNP, test_size = 1/5, random_state = 0 )
    # Split train into spam and ham
    ham = []
    spam = []
    for line in train:
        if(line.split()[0] == 'ham'):
            ham.append(line[4::1])
        else:
            spam.append(line[5::1])

    # Get list of all spam words
    spam_words = []
    for part in spam:
        split = part.split()
        for splity in split:
            spam_words.append(splity)
    
    # Get list of all ham words
    ham_words = []
    for part in ham:
        split = part.split()
        for splity in split:
            ham_words.append(splity)

    # Get list of all total words
    total_words = []
    for part in ham:
        split = part.split()
        for splity in split:
            total_words.append(splity)
    for part in spam:
        split = part.split()
        for splity in split:
            total_words.append(splity)
    unique_words = list(dict.fromkeys(total_words))

    # Calculate Values
    num_ham_messages = len(ham)
    num_spam_messages = len(spam)
    num_total_messages = num_ham_messages + num_spam_messages
    pSpam = num_spam_messages / num_total_messages
    pHam = num_ham_messages / num_total_messages

    num_total_words = len(total_words)
    num_spam_words = len(spam_words)
    num_ham_words = len(ham_words)

    AllLines = np.concatenate((ham, spam))

    alpha = 0   # Wasn't helpful so set to zero removes its use

    # P(w|spam) for all words
    wordProbs = {}
    # Solve for the constant P(w|spam) equation denominator
    denSum = 0
    denSum_noIDF = 0
    denSum_ham = 0
    denSum_ham_noIDF = 0
    theCount = 0
    theTotal = num_total_words
    for word in total_words:
        # Calculate IDF
        cntr = 0    # Number of messages with the word in it
        for line in AllLines:
            if word in line:
                cntr += 1
        IDF = math.log10(num_total_messages / cntr)
        # Calculate TF(w|spam)
        numWInSpamData = spam_words.count(word)
        TF = numWInSpamData / num_spam_words
        # Calculate TF(w|ham)
        numWInHamData = ham_words.count(word)
        TF_ham = numWInHamData / num_ham_words
        # Calculate pW
        pW = total_words.count(word) / num_total_words
        # Store info
        if word not in wordProbs:
            wordProbs[word] = [IDF, TF, pW, TF_ham]
        denSum += IDF * TF
        denSum_noIDF += TF
        denSum_ham += IDF * TF_ham
        denSum_ham_noIDF += TF_ham
        print("Status: " + str((theCount/theTotal) * 100)[0:5] + "%")
        theCount+=1
    denSum += alpha * num_spam_words
    denSum_ham += alpha * num_ham_words

        
    # for key, value in wordProbs.items():
    #     print(key + ": " + str(value[0]))

    numCorrect = 0
    numCorrect_noIDF = 0
    totalMs = len(test)
    numHamMs = 0
    numSpamMs = 0
    for testMessage in test:
        answer = ""
        if(testMessage.split()[0] == 'ham'):
            testMessage = testMessage[4::1]
            answer = "ham"
            numHamMs += 1
        else:
            testMessage = testMessage[5::1]
            answer = "spam"
            numSpamMs += 1

        testMessageWords = testMessage.split()

        pGSpamProd = 1
        for word in testMessageWords:
            IDF = 1
            TF = 1
            if word in wordProbs:
                IDF = wordProbs[word][0]
                TF = wordProbs[word][1]
            pWSpam = ( TF * IDF + alpha ) / denSum
            pGSpamProd *= pWSpam
        
        pGHamProd = 1
        for word in testMessageWords:
            IDF = 1
            TF_ham = 1
            if word in wordProbs:
                IDF = wordProbs[word][0]
                TF_ham = wordProbs[word][3]
            pWHam = ( TF_ham * IDF + alpha ) / denSum_ham
            pGHamProd *= pWHam

        pGSpamProd_noIDF = 1
        for word in testMessageWords:
            TF = 1
            if word in wordProbs:
                TF = wordProbs[word][1]
            pWSpam = ( TF + alpha ) / denSum_noIDF
            pGSpamProd_noIDF *= pWSpam
        
        pGHamProd_noIDF = 1
        for word in testMessageWords:
            TF_ham = 1
            if word in wordProbs:
                TF_ham = wordProbs[word][3]
            pWHam = ( TF_ham + alpha ) / denSum_ham
            pGHamProd_noIDF *= pWHam
        
        pWsProd = 1
        for word in testMessageWords:
            if word in wordProbs:
                pWsProd *= wordProbs[word][2]
        
        predSpam = ( pGSpamProd * pSpam )
        predHam = ( pGHamProd * pHam )
        result = ("ham" if predHam >= predSpam else "spam")
        numCorrect += (1 if result == answer else 0)

        predSpam_noIDF = ( pGSpamProd_noIDF * pSpam )
        predHam_noIDF = ( pGHamProd_noIDF * pHam )
        result_noIDF = ("ham" if predHam_noIDF >= predSpam_noIDF else "spam")
        numCorrect_noIDF += (1 if result_noIDF == answer else 0)

        # if(result != answer):
        #     print("Actually " + answer + ": " + testMessage)


    print("Train Data: " + str(num_ham_messages) + " ham messages & " + str(num_spam_messages) + " spam messages")
    print("Test Data: " + str(numHamMs) + " ham messages & " + str(numSpamMs) + " spam messages")

    print("Accuracy with Bag of Words: " + str(numCorrect_noIDF) + "/" + str(totalMs) + " = " + str((numCorrect_noIDF/totalMs)*100)[0:5] + "% Correct")
    print("Accuracy with using TF-IDF: " + str(numCorrect) + "/" + str(totalMs) + " = " + str((numCorrect/totalMs)*100)[0:5] + "% Correct\n")

    # Train Data: 3146 ham messages & 474 spam messages
    # Test Data: 1570 ham messages & 240 spam messages
    # Accuracy with Bag of Words: 1720/1810=95.02% Correct
    # Accuracy with using TF-IDF: 1722/1810=95.13% Correct
    
    



