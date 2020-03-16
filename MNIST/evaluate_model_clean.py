#!/usr/bin/env python
# coding: utf-8

# In[3]:


from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import numpy as np
from numpy import save
from numpy import asarray
from numpy import load
import sys
fil = sys.argv[1]

# run the test harness for evaluating adversarial model
def run_test_harness():
        # load adversarial dataset
        testX = load('clean_test_images.npy')
        testY = load('clean_test_labels.npy')
        model = load_model(fil)
        # evaluate model on adversarial test dataset
        loss, acc = model.evaluate(testX, testY, verbose=0)
        print('Accuracy for clean inputs> ',acc)
        print('Loss for clean inputs> ',loss)

# entry point, run the test harness
run_test_harness()


# In[ ]:




