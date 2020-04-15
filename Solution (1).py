#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras


# In[65]:


from tensorflow.keras.models import Model
classifier = keras.models.load_model("model_1_1_(5).h5")


# In[2]:


model = keras.models.load_model("model_1_1_(5).h5")


# In[4]:


model.summary()


# In[31]:


abs_weights = list(map(abs, model.get_weights()[0]))

from operator import add
from  functools import reduce

r_f = lambda x : reduce(add, x)
abs_weights_summed = list(map(r_f, abs_weights))
abs_weights_summed


# In[100]:


model.get_weights()[0].shape


# In[87]:


dir(model.layers[0])


# In[99]:


print(model.layers[0].filters)
print(model.layers[0].kernel_size)
print(model.layers[0].strides)
print(model.layers[0].padding)
print(model.layers[0].kernel_constraint)
print(model.layers[0].kernel_initializer)
print(model.layers[0].kernel_regularizer)
print(model.layers[0].kernel)


# In[149]:


model.get_weights()[0][:, :, 0]


# In[150]:


model.layers[0].bias[0]


# In[247]:


test = df.iloc[4]['ohe']
img_tensor = np.expand_dims(test, axis=0)
activations = activation_model.predict(img_tensor)
first_layer_activation = activations[0]
print(first_layer_activation.shape)


# In[160]:


df.iloc[4]['ohe'].shape


# In[248]:


first_layer_activation[0, :, 0] - model.layers[0].bias[0]


# In[249]:


big_boy = np.concatenate((np.zeros((3, 4)), test, np.zeros((4, 4))))
base = first_layer_activation[0, :, 0]
results = []
for i in range(8):
    multi = np.concatenate((np.zeros(7-i), base, np.zeros(i)))
    multiplied = np.multiply(big_boy.T, multi).T
    results.append(np.delete(multiplied, [0,1,2,203,204,205,206], 0)) # restore to previous size


# In[251]:


np.mean(results, axis=0)


# In[267]:


from PIL import Image
#checkout https://github.com/keplr-io/quiver
from matplotlib import cm
from sklearn.preprocessing import minmax_scale


def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height + 10))
    return dst


# In[95]:


import matplotlib.pyplot as plt
print(df.iloc[4]['raw'])
print(len(df.iloc[4]['raw']))
df.iloc[0]['ohe'].T
H = df.iloc[4]['ohe'].T  # added some commas and array creation code


# In[253]:


fig, ax = plt.subplots(figsize=(16, 400))
ax.get_yaxis().set_visible(False)
ax.imshow(np.mean(results, axis=0).T, interpolation='nearest')
plt.tight_layout()


# In[151]:


classes = classifier.predict_classes(np.array([df.iloc[4]['ohe'],]))
print("Predicted class is:",classes)


# In[60]:


classifier.input


# In[153]:


layer_outputs = [layer.output for layer in classifier.layers[:5]] 
activation_model = Model(inputs=classifier.input, outputs=layer_outputs)


# In[ ]:


plt.figure(figsize = (64,400))
im1 = plt.imshow(test.T, cmap="binary")
plt.axis('off')
plt.savefig("base.png", bbox_inches='tight')
plt.show()


# In[272]:


plt.figure(figsize = (16,200))
im1 = plt.imshow(test.T, cmap="hot")
plt.axis('off')
plt.savefig("base.png", bbox_inches='tight')
plt.show()

for j in range(16):
    big_boy = np.concatenate((np.zeros((3, 4)), test, np.zeros((4, 4))))
    base = first_layer_activation[0, :, j]
    results = []
    for i in range(8):
        multi = np.concatenate((np.zeros(7-i), base, np.zeros(i)))
        multiplied = np.multiply(big_boy.T, multi).T
        results.append(np.delete(multiplied, [0,1,2,203,204,205,206], 0)) # restore to previous size

    #plt.figure(figsize = (16,200))
    #im2 = plt.imshow(np.repeat(np.array([first_layer_activation[0, :, j]]), 4, axis=0), cmap="hot")
    #plt.axis('off')
    #plt.savefig("activation" + str(j) + ".png", bbox_inches='tight')
    #plt.show()

    plt.figure(figsize = (16,200))
    im3 = plt.imshow(np.mean(results, axis=0).T)
    
    plt.axis('off')
    plt.savefig("result" + str(j) + ".png", bbox_inches='tight')


# In[ ]:


final_img


# In[136]:


H2 = first_layer_activation[0, :, 1:2].T
H2 = 
H2.shape

H3 = np.multiply(H,H2[0])
H3.shape


# In[138]:


plt.figure(figsize = (64,200))
im2 = plt.imshow(H2[0], cmap="hot")
plt.show()
plt.figure(figsize = (64,200))
im1 = plt.imshow(H, interpolation='nearest')
plt.show()

plt.figure(figsize = (64,200))
im3 = plt.imshow(H3, interpolation='nearest')
plt.show()


# In[42]:


np.uint8(minmax_scale(abs_weights_summed[0])*255)


# In[30]:


img = None
for weights in abs_weights_summed[0]:
    if (img == None):
        img = Image.fromarray(np.uint8(minmax_scale(weights)*255))
    else:
        img = get_concat_v(img, Image.fromarray(np.uint8(minmax_scale(weights)*255)))


img.resize((img.size[0]*20,img.size[1]*20), Image.ANTIALIAS)


# In[7]:


import numpy as np
import pandas as pd
def sequence_to_ohe(
        sequence,
        channel={
            'A': 0,
            'T': 1,
            'U': 1,
            'C': 2,
            'G': 3
        }
):
    sequence_size = len(sequence)
    ohe_dataset = np.zeros((sequence_size, 4))

    for pos, nucleotide in enumerate(sequence):
        if nucleotide == 'N':
            continue
        ohe_dataset[pos, channel[nucleotide]] = 1
    return ohe_dataset


# In[3]:


raw_values = []
expected_results = []

with open('test_set_1_1.txt', 'r') as datafile:
    for line in datafile.readlines():
        a, b = line.split("\t")
        raw_values.append(a)
        if (b[0] == 'n'):
            expected_results.append(0)
        else:
            expected_results.append(1)


# In[21]:


print(raw_values[:5])
print(expected_results[:5])


# In[8]:


df = pd.DataFrame(list(zip(raw_values, expected_results)), 
               columns =['raw', 'expected'])

df.head()


# In[9]:


df['ohe'] = df.apply(lambda x: sequence_to_ohe(x['raw']), axis=1)


# In[40]:


df['predicted'] = df.apply(lambda x: model.predict(np.array([x['ohe']])), axis=1)


# In[47]:


df['predicted'] = df.apply(lambda x: x['predicted'][0][0], axis=1)


# In[48]:


df['predicted', 'expected']].head()


# In[50]:


df[['raw','predicted', 'expected']].to_csv("processed_data_backup.csv", index=False)


# In[4]:


import pandas as pd
df = pd.read_csv("processed_data_backup.csv")


# In[63]:


x = sequence_to_ohe(df.iloc[1]['raw'])
print(x.shape)
print(model.weights[0].shape)
model.layers[0].__call__(np.array([x]))


# In[5]:


df['diff'] = abs(df['expected'] - df['predicted'])


# In[6]:


df[['diff','predicted', 'expected']].head()


# In[7]:


sorted_df = df.sort_values(by='diff')
sorted_df[['diff','predicted', 'expected']].head()


# In[8]:


most_accurate = sorted_df.iloc[10000]
most_accurate


# In[17]:


len(most_accurate['raw'])


# In[9]:


change = []
position = []
predicted = []

raw = list(most_accurate['raw'])
nucl = ['A', 'T', 'C', 'G', 'N']
for i in range(120, 131):
    for letter in nucl:
        c_raw = raw.copy()
        if c_raw[i] == letter:
            continue
        else:
            c_raw[i] = letter
            change.append(letter)
            position.append(i)
            prediction = model.predict(np.array([sequence_to_ohe(c_raw)]))[0][0]
            predicted.append(prediction)
    


# In[10]:


changesdf = pd.DataFrame(list(zip(change, position, predicted)), 
               columns =['change', 'position', 'predicted'])
changesdf['diff'] = changesdf['predicted'] - most_accurate['predicted']
changesdf['abs_diff'] = abs(changesdf['diff'])
changesdf.sort_values(by='abs_diff', ascending=False)


# In[77]:


changesdf[['abs_diff','position']].groupby('position').mean().sort_values(by='abs_diff', ascending=False).plot.bar(figsize=(15,5), title="Mean absolute difference in prediction by positin on positions 120 - 130")


# In[51]:


puredf = changesdf_tens[['position','abs_diff','change']].sort_values(by='position')


# In[30]:


puredf.pivot(columns='position', values='abs_diff', index='change').loc[:, 0:10].plot.box()


# In[69]:


puredf.groupby('position').var().sort_values(by='abs_diff', ascending=False).iloc[:30].plot.bar(figsize=(15,5), title="Variance in absolute difference in prediction by starting position")


# In[62]:


puredf[['abs_diff','change']].groupby('change').mean().sort_values(by='abs_diff', ascending=False).plot.bar(figsize=(15,5), title="Mean absolute difference in prediction by change sequence")


# In[63]:


puredf[['abs_diff','change']].groupby('change').max().sort_values(by='abs_diff', ascending=False).plot.bar(figsize=(15,5), title="Max absolute difference in prediction by change sequence")


# In[66]:


puredf[['abs_diff','change']].groupby('change').var().sort_values(by='abs_diff', ascending=False).plot.bar(figsize=(15,5), title="Variance in absolute difference in prediction by change sequence")


# In[12]:


import random
random.randint(0, len(nucl))


# In[13]:


tens = [
    "N"*10,
    "T"*10,
    "C"*10,
    "A"*10,
    "G"*10
]
for i in range(10):
    tens.append(''.join([nucl[random.randint(0, 4)] for j in range(10)]))
tens


# In[14]:


change = []
position = []
predicted = []

raw = list(most_accurate['raw'])
for i in range(len(raw)-10):
    for ten in tens:
        c_raw = raw.copy()
        for j in range(10):
            c_raw[i+j] = ten[j]
        change.append(ten)
        position.append(i)
        prediction = model.predict(np.array([sequence_to_ohe(c_raw)]))[0][0]
        predicted.append(prediction)
    
changesdf_pairs = pd.DataFrame(list(zip(change, position, predicted)), 
               columns =['change', 'position', 'predicted'])
changesdf_pairs['diff'] = changesdf_pairs['predicted'] - most_accurate['predicted']
changesdf_pairs['abs_diff'] = abs(changesdf_pairs['diff'])


# In[15]:


changesdf_pairs.sort_values(by='diff', ascending=False).iloc[:50]
changesdf_tens = changesdf_pairs
changesdf_tens

