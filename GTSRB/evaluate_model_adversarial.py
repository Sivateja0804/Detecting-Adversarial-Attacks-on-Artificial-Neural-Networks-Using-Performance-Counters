#!/usr/bin/env python
# coding: utf-8

# In[3]:


from tensorflow.keras.models import load_model
import numpy as np
from numpy import save
from numpy import asarray
from numpy import load
#from profiler import *
import sys

fil = sys.argv[1]

#try:
#    events= [['PERF_COUNT_HW_INSTRUCTIONS'], ['PERF_COUNT_HW_BRANCH_INSTRUCTIONS'], ['PERF_COUNT_HW_BRANCH_MISSES'], ['PERF_COUNT_SW_PAGE_FAULTS'], ['L1-ICACHE-LOAD-MISSES'], ['LLC-LOAD-MISSES'], ['LLC-STORE-MISSES'], ['ITLB-LOAD-MISSES'], ['DTLB-LOAD-MISSES']]
#    perf= Profiler(program_args= ['/bin/ls','/'], events_groups=events)
#    data= perf.run(sample_period= 0.01)
#except RuntimeError as e:
#    print(e.args[0])
# run the test harness for evaluating adversarial model
def run_test_harness():
        # load adversarial dataset
        adv_testX = load('adversarial_test_images.npy')
        adv_testY = load('adversarial_test_labels.npy')
        model = load_model(fil)
        # evaluate model on adversarial test dataset
        loss, acc = model.evaluate(adv_testX, adv_testY, verbose=0)
        print('Accuracy for adversarial inputs> ',acc)
        print('Loss for adversarial inputs> ',loss)
        #print(data)

# entry point, run the test harness
run_test_harness()


# In[ ]:




