
# Improvise a Jazz Solo with an LSTM Network

Welcome to your final programming assignment of this week! In this notebook, you will implement a model that uses an LSTM to generate music. You will even be able to listen to your own music at the end of the assignment. 

**You will learn to:**
- Apply an LSTM to music generation.
- Generate your own jazz music with deep learning.


## <font color='darkblue'>Updates</font>

#### If you were working on the notebook before this update...
* The current notebook is version "3a".
* You can find your original work saved in the notebook with the previous version name ("v3") 
* To view the file directory, go to the menu "File->Open", and this will open a new tab that shows the file directory.

#### List of updates
* `djmodel`
    - Explains `Input` layer and its parameter `shape`.
    - Explains `Lambda` layer and replaces the given solution with hints and sample code (to improve the learning experience).
    - Adds hints for using the Keras `Model`.
* `music_inference_model`
    - Explains each line of code in the `one_hot` function.
    - Explains how to apply `one_hot` with a Lambda layer instead of giving the code solution (to improve the learning experience).
    - Adds instructions on defining the `Model`.
* `predict_and_sample`
    - Provides detailed instructions for each step.
    - Clarifies which variable/function to use for inference.
* Spelling, grammar and wording corrections.

Please run the following cell to load all the packages required in this assignment. This may take a few minutes. 


```python
from __future__ import print_function
import IPython
import sys
from music21 import *
import numpy as np
from grammar import *
from qa import *
from preprocess import * 
from music_utils import *
from data_utils import *
from keras.models import load_model, Model
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from keras.initializers import glorot_uniform
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras import backend as K
```

    Using TensorFlow backend.


## 1 - Problem statement

You would like to create a jazz music piece specially for a friend's birthday. However, you don't know any instruments or music composition. Fortunately, you know deep learning and will solve this problem using an LSTM network.  

You will train a network to generate novel jazz solos in a style representative of a body of performed work.

<img src="images/jazz.jpg" style="width:450;height:300px;">


### 1.1 - Dataset

You will train your algorithm on a corpus of Jazz music. Run the cell below to listen to a snippet of the audio from the training set:


```python
IPython.display.Audio('./data/30s_seq.mp3')
```





    




We have taken care of the preprocessing of the musical data to render it in terms of musical "values." 

#### Details about music (optional)
You can informally think of each "value" as a note, which comprises a pitch and duration. For example, if you press down a specific piano key for 0.5 seconds, then you have just played a note. In music theory, a "value" is actually more complicated than this--specifically, it also captures the information needed to play multiple notes at the same time. For example, when playing a music piece, you might press down two piano keys at the same time (playing multiple notes at the same time generates what's called a "chord"). But we don't need to worry about the details of music theory for this assignment. 

#### Music as a sequence of values
* For the purpose of this assignment, all you need to know is that we will obtain a dataset of values, and will learn an RNN model to generate sequences of values. 
* Our music generation system will use 78 unique values. 

Run the following code to load the raw music data and preprocess it into values. This might take a few minutes.


```python
X, Y, n_values, indices_values = load_music_utils()
print('number of training examples:', X.shape[0])
print('Tx (length of sequence):', X.shape[1])
print('total # of unique values:', n_values)
print('shape of X:', X.shape)
print('Shape of Y:', Y.shape)
```

    number of training examples: 60
    Tx (length of sequence): 30
    total # of unique values: 78
    shape of X: (60, 30, 78)
    Shape of Y: (30, 60, 78)


You have just loaded the following:

- `X`: This is an (m, $T_x$, 78) dimensional array. 
    - We have m training examples, each of which is a snippet of $T_x =30$ musical values. 
    - At each time step, the input is one of 78 different possible values, represented as a one-hot vector. 
        - For example, X[i,t,:] is a one-hot vector representing the value of the i-th example at time t. 

- `Y`: a $(T_y, m, 78)$ dimensional array
    - This is essentially the same as `X`, but shifted one step to the left (to the past). 
    - Notice that the data in `Y` is **reordered** to be dimension $(T_y, m, 78)$, where $T_y = T_x$. This format makes it more convenient to feed into the LSTM later.
    - Similar to the dinosaur assignment, we're using the previous values to predict the next value.
        - So our sequence model will try to predict $y^{\langle t \rangle}$ given $x^{\langle 1\rangle}, \ldots, x^{\langle t \rangle}$. 

- `n_values`: The number of unique values in this dataset. This should be 78. 

- `indices_values`: python dictionary mapping integers 0 through 77 to musical values.

### 1.2 - Overview of our model

Here is the architecture of the model we will use. This is similar to the Dinosaurus model, except that you will implement it in Keras.

<img src="images/music_generation.png" style="width:600;height:400px;">


* $X = (x^{\langle 1 \rangle}, x^{\langle 2 \rangle}, \cdots, x^{\langle T_x \rangle})$ is a window of size $T_x$ scanned over the musical corpus. 
* Each $x^{\langle t \rangle}$ is an index corresponding to a value.
* $\hat{y}^{t}$ is the prediction for the next value.
* We will be training the model on random snippets of 30 values taken from a much longer piece of music. 
    - Thus, we won't bother to set the first input $x^{\langle 1 \rangle} = \vec{0}$, since most of these snippets of audio start somewhere in the middle of a piece of music. 
    - We are setting each of the snippets to have the same length $T_x = 30$ to make vectorization easier.

## Overview of parts 2 and 3

* We're going to train a model that predicts the next note in a style that is similar to the jazz music that it's trained on.  The training is contained in the weights and biases of the model. 
* In Part 3, we're then going to use those weights and biases in a new model which predicts a series of notes, using the previous note to predict the next note. 
* The weights and biases are transferred to the new model using 'global shared layers' described below"


## 2 - Building the model

* In this part you will build and train a model that will learn musical patterns. 
* The model takes input X of shape $(m, T_x, 78)$ and labels Y of shape $(T_y, m, 78)$. 
* We will use an LSTM with hidden states that have $n_{a} = 64$ dimensions.


```python
# number of dimensions for the hidden state of each LSTM cell.
n_a = 64 
```


#### Sequence generation uses a for-loop
* If you're building an RNN where, at test time, the entire input sequence $x^{\langle 1 \rangle}, x^{\langle 2 \rangle}, \ldots, x^{\langle T_x \rangle}$ is given in advance, then Keras has simple built-in functions to build the model. 
* However, for **sequence generation, at test time we don't know all the values of $x^{\langle t\rangle}$ in advance**.
* Instead we generate them one at a time using $x^{\langle t\rangle} = y^{\langle t-1 \rangle}$. 
    * The input at time "t" is the prediction at the previous time step "t-1".
* So you'll need to implement your own for-loop to iterate over the time steps. 

#### Shareable weights
* The function `djmodel()` will call the LSTM layer $T_x$ times using a for-loop.
* It is important that all $T_x$ copies have the same weights. 
    - The $T_x$ steps should have shared weights that aren't re-initialized.
* Referencing a globally defined shared layer will utilize the same layer-object instance at each time step.
* The key steps for implementing layers with shareable weights in Keras are: 
1. Define the layer objects (we will use global variables for this).
2. Call these objects when propagating the input.

#### 3 types of layers
* We have defined the layers objects you need as global variables.  
* Please run the next cell to create them. 
* Please read the Keras documentation and understand these layers: 
    - [Reshape()](https://keras.io/layers/core/#reshape): Reshapes an output to a certain shape.
    - [LSTM()](https://keras.io/layers/recurrent/#lstm): Long Short-Term Memory layer
    - [Dense()](https://keras.io/layers/core/#dense): A regular fully-connected neural network layer.



```python
n_values = 78 # number of music values
reshapor = Reshape((1, n_values))                        # Used in Step 2.B of djmodel(), below
LSTM_cell = LSTM(n_a, return_state = True)         # Used in Step 2.C
densor = Dense(n_values, activation='softmax')     # Used in Step 2.D
```

* `reshapor`, `LSTM_cell` and `densor` are globally defined layer objects, that you'll use to implement `djmodel()`. 
* In order to propagate a Keras tensor object X through one of these layers, use `layer_object()`.
    - For one input, use `layer_object(X)`
    - For more than one input, put the inputs in a list: `layer_object([X1,X2])`

 **Exercise**: Implement `djmodel()`. 

#### Inputs (given)
* The `Input()` layer is used for defining the input `X` as well as the initial hidden state 'a0' and cell state `c0`.
* The `shape` parameter takes a tuple that does not include the batch dimension (`m`).
    - For example,
    ```Python
    X = Input(shape=(Tx, n_values)) # X has 3 dimensions and not 2: (m, Tx, n_values)
    ```
#### Step 1: Outputs (TODO)
1. Create an empty list "outputs" to save the outputs of the LSTM Cell at every time step.

#### Step 2: Loop through time steps (TODO)
* Loop for $t \in 1, \ldots, T_x$:

#### 2A. Select the 't' time-step vector from X.
* X has the shape (m, Tx, n_values).
* The shape of the 't' selection should be (n_values,). 
* Recall that if you were implementing in numpy instead of Keras, you would extract a slice from a 3D numpy array like this:
```Python
var1 = array1[:,1,:]
```

#### Lambda layer
* Since we're using Keras, we need to define this step inside a custom layer.
* In Keras, this is a Lambda layer [Lambda](https://keras.io/layers/core/#lambda)
* As an example, a Lambda layer that takes the previous layer and adds '1' looks like this
```    
       lambda_layer1 = Lambda(lambda z: z + 1)(previous_layer)
```
* The previous layer in this case is `X`.
* `z` is a local variable of the lambda function. 
    * The `previous_layer` gets passed into the parameter `z` in the lowercase `lambda` function.
    * You can choose the name of the variable to be something else if you want.
* The operation after the colon ':' should be the operation to extract a slice from the previous layer.
* **Hint**: You'll be using the variable `t` within the definition of the lambda layer even though it isn't passed in as an argument to Lambda.

#### 2B. Reshape x to be (1,n_values).
* Use the `reshapor()` layer.  It is a function that takes the previous layer as its input argument.

#### 2C. Run x through one step of LSTM_cell.
* Initialize the `LSTM_cell` with the previous step's hidden state $a$ and cell state $c$. 
* Use the following formatting:
```python
next_hidden_state, _, next_cell_state = LSTM_cell(inputs=input_x, initial_state=[previous_hidden_state, previous_cell_state])
```
    * Choose appropriate variables for inputs, hidden state and cell state.

#### 2D. Dense layer
* Propagate the LSTM's hidden state through a dense+softmax layer using `densor`. 
  
#### 2E. Append output
* Append the output to the list of "outputs".


#### Step 3: After the loop, create the model
* Use the Keras `Model` object to create a model.
* specify the inputs and outputs:
```Python
model = Model(inputs=[input_x, initial_hidden_state, initial_cell_state], outputs=the_outputs)
```
    * Choose the appropriate variables for the input tensor, hidden state, cell state, and output.
* See the documentation for [Model](https://keras.io/models/model/)


```python
# GRADED FUNCTION: djmodel

def djmodel(Tx, n_a, n_values):
    """
    Implement the model
    
    Arguments:
    Tx -- length of the sequence in a corpus
    n_a -- the number of activations used in our model
    n_values -- number of unique values in the music data 
    
    Returns:
    model -- a keras instance model with n_a activations
    """
    
    # Define the input layer and specify the shape
    X = Input(shape=(Tx, n_values))
    
    # Define the initial hidden state a0 and initial cell state c0
    # using `Input`
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0
    
    ### START CODE HERE ### 
    # Step 1: Create empty list to append the outputs while you iterate (≈1 line)
    outputs = []
    
    # Step 2: Loop
    for t in range(Tx):
        
        # Step 2.A: select the "t"th time step vector from X. 
        x = Lambda(lambda z: z[:,t,:])(X)
        # Step 2.B: Use reshapor to reshape x to be (1, n_values) (≈1 line)
        x = reshapor(x)
        # Step 2.C: Perform one step of the LSTM_cell
        a, _, c = LSTM_cell(inputs=x, initial_state=[a, c])
        # Step 2.D: Apply densor to the hidden state output of LSTM_Cell
        out = densor(a)
        # Step 2.E: add the output to "outputs"
        outputs.append(out)
        
    # Step 3: Create model instance
    model = Model(inputs=[X, a0, c0], outputs=outputs)
    
    ### END CODE HERE ###
    
    return model
```

#### Create the model object
* Run the following cell to define your model. 
* We will use `Tx=30`, `n_a=64` (the dimension of the LSTM activations), and `n_values=78`. 
* This cell may take a few seconds to run. 


```python
model = djmodel(Tx = 30 , n_a = 64, n_values = 78)
```


```python
# Check your model
model.summary()
```

    ____________________________________________________________________________________________________
    Layer (type)                     Output Shape          Param #     Connected to                     
    ====================================================================================================
    input_5 (InputLayer)             (None, 30, 78)        0                                            
    ____________________________________________________________________________________________________
    lambda_1 (Lambda)                (None, 78)            0           input_5[0][0]                    
    ____________________________________________________________________________________________________
    reshape_1 (Reshape)              (None, 1, 78)         0           lambda_1[0][0]                   
                                                                       lambda_2[0][0]                   
                                                                       lambda_3[0][0]                   
                                                                       lambda_4[0][0]                   
                                                                       lambda_5[0][0]                   
                                                                       lambda_6[0][0]                   
                                                                       lambda_7[0][0]                   
                                                                       lambda_8[0][0]                   
                                                                       lambda_9[0][0]                   
                                                                       lambda_10[0][0]                  
                                                                       lambda_11[0][0]                  
                                                                       lambda_12[0][0]                  
                                                                       lambda_13[0][0]                  
                                                                       lambda_14[0][0]                  
                                                                       lambda_15[0][0]                  
                                                                       lambda_16[0][0]                  
                                                                       lambda_17[0][0]                  
                                                                       lambda_18[0][0]                  
                                                                       lambda_19[0][0]                  
                                                                       lambda_20[0][0]                  
                                                                       lambda_21[0][0]                  
                                                                       lambda_22[0][0]                  
                                                                       lambda_23[0][0]                  
                                                                       lambda_24[0][0]                  
                                                                       lambda_25[0][0]                  
                                                                       lambda_26[0][0]                  
                                                                       lambda_27[0][0]                  
                                                                       lambda_28[0][0]                  
                                                                       lambda_29[0][0]                  
                                                                       lambda_30[0][0]                  
    ____________________________________________________________________________________________________
    a0 (InputLayer)                  (None, 64)            0                                            
    ____________________________________________________________________________________________________
    c0 (InputLayer)                  (None, 64)            0                                            
    ____________________________________________________________________________________________________
    lambda_2 (Lambda)                (None, 78)            0           input_5[0][0]                    
    ____________________________________________________________________________________________________
    lstm_1 (LSTM)                    [(None, 64), (None, 6 36608       reshape_1[90][0]                 
                                                                       a0[0][0]                         
                                                                       c0[0][0]                         
                                                                       reshape_1[91][0]                 
                                                                       lstm_1[90][0]                    
                                                                       lstm_1[90][2]                    
                                                                       reshape_1[92][0]                 
                                                                       lstm_1[91][0]                    
                                                                       lstm_1[91][2]                    
                                                                       reshape_1[93][0]                 
                                                                       lstm_1[92][0]                    
                                                                       lstm_1[92][2]                    
                                                                       reshape_1[94][0]                 
                                                                       lstm_1[93][0]                    
                                                                       lstm_1[93][2]                    
                                                                       reshape_1[95][0]                 
                                                                       lstm_1[94][0]                    
                                                                       lstm_1[94][2]                    
                                                                       reshape_1[96][0]                 
                                                                       lstm_1[95][0]                    
                                                                       lstm_1[95][2]                    
                                                                       reshape_1[97][0]                 
                                                                       lstm_1[96][0]                    
                                                                       lstm_1[96][2]                    
                                                                       reshape_1[98][0]                 
                                                                       lstm_1[97][0]                    
                                                                       lstm_1[97][2]                    
                                                                       reshape_1[99][0]                 
                                                                       lstm_1[98][0]                    
                                                                       lstm_1[98][2]                    
                                                                       reshape_1[100][0]                
                                                                       lstm_1[99][0]                    
                                                                       lstm_1[99][2]                    
                                                                       reshape_1[101][0]                
                                                                       lstm_1[100][0]                   
                                                                       lstm_1[100][2]                   
                                                                       reshape_1[102][0]                
                                                                       lstm_1[101][0]                   
                                                                       lstm_1[101][2]                   
                                                                       reshape_1[103][0]                
                                                                       lstm_1[102][0]                   
                                                                       lstm_1[102][2]                   
                                                                       reshape_1[104][0]                
                                                                       lstm_1[103][0]                   
                                                                       lstm_1[103][2]                   
                                                                       reshape_1[105][0]                
                                                                       lstm_1[104][0]                   
                                                                       lstm_1[104][2]                   
                                                                       reshape_1[106][0]                
                                                                       lstm_1[105][0]                   
                                                                       lstm_1[105][2]                   
                                                                       reshape_1[107][0]                
                                                                       lstm_1[106][0]                   
                                                                       lstm_1[106][2]                   
                                                                       reshape_1[108][0]                
                                                                       lstm_1[107][0]                   
                                                                       lstm_1[107][2]                   
                                                                       reshape_1[109][0]                
                                                                       lstm_1[108][0]                   
                                                                       lstm_1[108][2]                   
                                                                       reshape_1[110][0]                
                                                                       lstm_1[109][0]                   
                                                                       lstm_1[109][2]                   
                                                                       reshape_1[111][0]                
                                                                       lstm_1[110][0]                   
                                                                       lstm_1[110][2]                   
                                                                       reshape_1[112][0]                
                                                                       lstm_1[111][0]                   
                                                                       lstm_1[111][2]                   
                                                                       reshape_1[113][0]                
                                                                       lstm_1[112][0]                   
                                                                       lstm_1[112][2]                   
                                                                       reshape_1[114][0]                
                                                                       lstm_1[113][0]                   
                                                                       lstm_1[113][2]                   
                                                                       reshape_1[115][0]                
                                                                       lstm_1[114][0]                   
                                                                       lstm_1[114][2]                   
                                                                       reshape_1[116][0]                
                                                                       lstm_1[115][0]                   
                                                                       lstm_1[115][2]                   
                                                                       reshape_1[117][0]                
                                                                       lstm_1[116][0]                   
                                                                       lstm_1[116][2]                   
                                                                       reshape_1[118][0]                
                                                                       lstm_1[117][0]                   
                                                                       lstm_1[117][2]                   
                                                                       reshape_1[119][0]                
                                                                       lstm_1[118][0]                   
                                                                       lstm_1[118][2]                   
    ____________________________________________________________________________________________________
    lambda_3 (Lambda)                (None, 78)            0           input_5[0][0]                    
    ____________________________________________________________________________________________________
    lambda_4 (Lambda)                (None, 78)            0           input_5[0][0]                    
    ____________________________________________________________________________________________________
    lambda_5 (Lambda)                (None, 78)            0           input_5[0][0]                    
    ____________________________________________________________________________________________________
    lambda_6 (Lambda)                (None, 78)            0           input_5[0][0]                    
    ____________________________________________________________________________________________________
    lambda_7 (Lambda)                (None, 78)            0           input_5[0][0]                    
    ____________________________________________________________________________________________________
    lambda_8 (Lambda)                (None, 78)            0           input_5[0][0]                    
    ____________________________________________________________________________________________________
    lambda_9 (Lambda)                (None, 78)            0           input_5[0][0]                    
    ____________________________________________________________________________________________________
    lambda_10 (Lambda)               (None, 78)            0           input_5[0][0]                    
    ____________________________________________________________________________________________________
    lambda_11 (Lambda)               (None, 78)            0           input_5[0][0]                    
    ____________________________________________________________________________________________________
    lambda_12 (Lambda)               (None, 78)            0           input_5[0][0]                    
    ____________________________________________________________________________________________________
    lambda_13 (Lambda)               (None, 78)            0           input_5[0][0]                    
    ____________________________________________________________________________________________________
    lambda_14 (Lambda)               (None, 78)            0           input_5[0][0]                    
    ____________________________________________________________________________________________________
    lambda_15 (Lambda)               (None, 78)            0           input_5[0][0]                    
    ____________________________________________________________________________________________________
    lambda_16 (Lambda)               (None, 78)            0           input_5[0][0]                    
    ____________________________________________________________________________________________________
    lambda_17 (Lambda)               (None, 78)            0           input_5[0][0]                    
    ____________________________________________________________________________________________________
    lambda_18 (Lambda)               (None, 78)            0           input_5[0][0]                    
    ____________________________________________________________________________________________________
    lambda_19 (Lambda)               (None, 78)            0           input_5[0][0]                    
    ____________________________________________________________________________________________________
    lambda_20 (Lambda)               (None, 78)            0           input_5[0][0]                    
    ____________________________________________________________________________________________________
    lambda_21 (Lambda)               (None, 78)            0           input_5[0][0]                    
    ____________________________________________________________________________________________________
    lambda_22 (Lambda)               (None, 78)            0           input_5[0][0]                    
    ____________________________________________________________________________________________________
    lambda_23 (Lambda)               (None, 78)            0           input_5[0][0]                    
    ____________________________________________________________________________________________________
    lambda_24 (Lambda)               (None, 78)            0           input_5[0][0]                    
    ____________________________________________________________________________________________________
    lambda_25 (Lambda)               (None, 78)            0           input_5[0][0]                    
    ____________________________________________________________________________________________________
    lambda_26 (Lambda)               (None, 78)            0           input_5[0][0]                    
    ____________________________________________________________________________________________________
    lambda_27 (Lambda)               (None, 78)            0           input_5[0][0]                    
    ____________________________________________________________________________________________________
    lambda_28 (Lambda)               (None, 78)            0           input_5[0][0]                    
    ____________________________________________________________________________________________________
    lambda_29 (Lambda)               (None, 78)            0           input_5[0][0]                    
    ____________________________________________________________________________________________________
    lambda_30 (Lambda)               (None, 78)            0           input_5[0][0]                    
    ____________________________________________________________________________________________________
    dense_1 (Dense)                  (None, 78)            5070        lstm_1[90][0]                    
                                                                       lstm_1[91][0]                    
                                                                       lstm_1[92][0]                    
                                                                       lstm_1[93][0]                    
                                                                       lstm_1[94][0]                    
                                                                       lstm_1[95][0]                    
                                                                       lstm_1[96][0]                    
                                                                       lstm_1[97][0]                    
                                                                       lstm_1[98][0]                    
                                                                       lstm_1[99][0]                    
                                                                       lstm_1[100][0]                   
                                                                       lstm_1[101][0]                   
                                                                       lstm_1[102][0]                   
                                                                       lstm_1[103][0]                   
                                                                       lstm_1[104][0]                   
                                                                       lstm_1[105][0]                   
                                                                       lstm_1[106][0]                   
                                                                       lstm_1[107][0]                   
                                                                       lstm_1[108][0]                   
                                                                       lstm_1[109][0]                   
                                                                       lstm_1[110][0]                   
                                                                       lstm_1[111][0]                   
                                                                       lstm_1[112][0]                   
                                                                       lstm_1[113][0]                   
                                                                       lstm_1[114][0]                   
                                                                       lstm_1[115][0]                   
                                                                       lstm_1[116][0]                   
                                                                       lstm_1[117][0]                   
                                                                       lstm_1[118][0]                   
                                                                       lstm_1[119][0]                   
    ====================================================================================================
    Total params: 41,678
    Trainable params: 41,678
    Non-trainable params: 0
    ____________________________________________________________________________________________________


**Expected Output**  
Scroll to the bottom of the output, and you'll see the following:

```Python
Total params: 41,678
Trainable params: 41,678
Non-trainable params: 0
```

#### Compile the model for training
* You now need to compile your model to be trained. 
* We will use:
    - optimizer: Adam optimizer
    - Loss function: categorical cross-entropy (for multi-class classification)


```python
opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
```

#### Initialize hidden state and cell state
Finally, let's initialize `a0` and `c0` for the LSTM's initial state to be zero. 


```python
m = 60
a0 = np.zeros((m, n_a))
c0 = np.zeros((m, n_a))
```

#### Train the model
* Lets now fit the model! 
* We will turn `Y` into a list, since the cost function expects `Y` to be provided in this format 
    - `list(Y)` is a list with 30 items, where each of the list items is of shape (60,78). 
    - Lets train for 100 epochs. This will take a few minutes. 


```python
model.fit([X, a0, c0], list(Y), epochs=100)
```

    





    <keras.callbacks.History at 0x7f39fe43c978>



#### Expected Output

The model loss will start high, (100 or so), and after 100 epochs, it should be in the single digits.  These won't be the exact number that you'll see, due to random initialization of weights.  
For example:
```
Epoch 1/100
60/60 [==============================] - 3s - loss: 125.7673
...
```
Scroll to the bottom to check Epoch 100
```
...
Epoch 100/100
60/60 [==============================] - 0s - loss: 6.1861
```

Now that you have trained a model, let's go to the final section to implement an inference algorithm, and generate some music! 

## 3 - Generating music

You now have a trained model which has learned the patterns of the jazz soloist. Lets now use this model to synthesize new music. 

#### 3.1 - Predicting & Sampling

<img src="images/music_gen.png" style="width:600;height:400px;">

At each step of sampling, you will:
* Take as input the activation '`a`' and cell state '`c`' from the previous state of the LSTM.
* Forward propagate by one step.
* Get a new output activation as well as cell state. 
* The new activation '`a`' can then be used to generate the output using the fully connected layer, `densor`. 

##### Initialization
* We will initialize the following to be zeros:
    * `x0` 
    * hidden state `a0` 
    * cell state `c0` 

**Exercise:** 
* Implement the function below to sample a sequence of musical values. 
* Here are some of the key steps you'll need to implement inside the for-loop that generates the $T_y$ output characters: 

* Step 2.A: Use `LSTM_Cell`, which takes in the input layer, as well as the previous step's '`c`' and '`a`' to generate the current step's '`c`' and '`a`'. 
```Python
next_hidden_state, _, next_cell_state = LSTM_cell(input_x, initial_state=[previous_hidden_state, previous_cell_state])
```
    * Choose the appropriate variables for the input_x, hidden_state, and cell_state

* Step 2.B: Compute the output by applying `densor` to compute a softmax on '`a`' to get the output for the current step. 

* Step 2.C: Append the output to the list `outputs`.



* Step 2.D: Sample x to be the one-hot version of '`out`'. 
* This allows you to pass it to the next LSTM's step.  
* We have provided the definition of `one_hot(x)` in the 'music_utils.py' file and imported it.
Here is the definition of `one_hot`
```Python
def one_hot(x):
    x = K.argmax(x)
    x = tf.one_hot(indices=x, depth=78) 
    x = RepeatVector(1)(x)
    return x
```
Here is what the `one_hot` function is doing:
* argmax: within the vector `x`, find the position with the maximum value and return the index of that position.  
    * For example: argmax of [-1,0,1] finds that 1 is the maximum value, and returns the index position, which is 2.  Read the documentation for [keras.argmax](https://www.tensorflow.org/api_docs/python/tf/keras/backend/argmax).
* one_hot: takes a list of indices and the depth of the one-hot vector (number of categories, which is 78 in this assignment).  It converts each index into the one-hot vector representation.  For instance, if the indices is [2], and the depth is 5, then the one-hot vector returned is [0,0,1,0,0].  Check out the documentation for [tf.one_hot](https://www.tensorflow.org/api_docs/python/tf/one_hot) for more examples and explanations.
* RepeatVector(n): This takes a vector and duplicates it `n` times.  Notice that we had it repeat 1 time.  This may seem like it's not doing anything.  If you look at the documentation for [RepeatVector](https://keras.io/layers/core/#repeatvector), you'll notice that if x is a vector with dimension (m,5) and it gets passed into `RepeatVector(1)`, then the output is (m,1,5).  In other words, it adds an additional dimension (of length 1) to the resulting vector.
* Apply the custom one_hot encoding using the [Lambda](https://keras.io/layers/core/#lambda) layer.  You saw earlier that the Lambda layer can be used like this:
```Python
result = Lambda(lambda x: x + 1)(input_var)
```

If you pre-define a function, you can do the same thing:
```Python
def add_one(x)
    return x + 1

# use the add_one function inside of the Lambda function
result = Lambda(add_one)(input_var)
```

#### Step 3: Inference Model:  
This is how to use the Keras `Model`.
```Python
model = Model(inputs=[input_x, initial_hidden_state, initial_cell_state], outputs=the_outputs)
```


* Choose the appropriate variables for the input tensor, hidden state, cell state, and output.
* **Hint**: the inputs to the model are the **initial** inputs and states.


```python
# GRADED FUNCTION: music_inference_model

def music_inference_model(LSTM_cell, densor, n_values = 78, n_a = 64, Ty = 100):
    """
    Uses the trained "LSTM_cell" and "densor" from model() to generate a sequence of values.
    
    Arguments:
    LSTM_cell -- the trained "LSTM_cell" from model(), Keras layer object
    densor -- the trained "densor" from model(), Keras layer object
    n_values -- integer, number of unique values
    n_a -- number of units in the LSTM_cell
    Ty -- integer, number of time steps to generate
    
    Returns:
    inference_model -- Keras model instance
    """
    
    # Define the input of your model with a shape 
    x0 = Input(shape=(1, n_values))
    
    # Define s0, initial hidden state for the decoder LSTM
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0
    x = x0

    ### START CODE HERE ###
    # Step 1: Create an empty list of "outputs" to later store your predicted values (≈1 line)
    outputs = []
    
    # Step 2: Loop over Ty and generate a value at every time step
    for t in range(Ty):
        
        # Step 2.A: Perform one step of LSTM_cell (≈1 line)
        a, _, c = LSTM_cell(x, initial_state=[a, c])
        
        # Step 2.B: Apply Dense layer to the hidden state output of the LSTM_cell (≈1 line)
        out = densor(a)

        # Step 2.C: Append the prediction "out" to "outputs". out.shape = (None, 78) (≈1 line)
        outputs.append(out)
        
        # Step 2.D: 
        # Select the next value according to "out",
        # Set "x" to be the one-hot representation of the selected value
        # See instructions above.
        x = Lambda(one_hot)(out)
        
    # Step 3: Create model instance with the correct "inputs" and "outputs" (≈1 line)
    inference_model = Model(inputs=[x0, a0, c0], outputs=outputs)
    
    ### END CODE HERE ###
    
    return inference_model
```

Run the cell below to define your inference model. This model is hard coded to generate 50 values.


```python
inference_model = music_inference_model(LSTM_cell, densor, n_values = 78, n_a = 64, Ty = 50)
```


```python
# Check the inference model
inference_model.summary()
```

    ____________________________________________________________________________________________________
    Layer (type)                     Output Shape          Param #     Connected to                     
    ====================================================================================================
    input_8 (InputLayer)             (None, 1, 78)         0                                            
    ____________________________________________________________________________________________________
    a0 (InputLayer)                  (None, 64)            0                                            
    ____________________________________________________________________________________________________
    c0 (InputLayer)                  (None, 64)            0                                            
    ____________________________________________________________________________________________________
    lstm_1 (LSTM)                    [(None, 64), (None, 6 36608       input_8[0][0]                    
                                                                       a0[0][0]                         
                                                                       c0[0][0]                         
                                                                       lambda_33[0][0]                  
                                                                       lstm_1[122][0]                   
                                                                       lstm_1[122][2]                   
                                                                       lambda_34[0][0]                  
                                                                       lstm_1[123][0]                   
                                                                       lstm_1[123][2]                   
                                                                       lambda_35[0][0]                  
                                                                       lstm_1[124][0]                   
                                                                       lstm_1[124][2]                   
                                                                       lambda_36[0][0]                  
                                                                       lstm_1[125][0]                   
                                                                       lstm_1[125][2]                   
                                                                       lambda_37[0][0]                  
                                                                       lstm_1[126][0]                   
                                                                       lstm_1[126][2]                   
                                                                       lambda_38[0][0]                  
                                                                       lstm_1[127][0]                   
                                                                       lstm_1[127][2]                   
                                                                       lambda_39[0][0]                  
                                                                       lstm_1[128][0]                   
                                                                       lstm_1[128][2]                   
                                                                       lambda_40[0][0]                  
                                                                       lstm_1[129][0]                   
                                                                       lstm_1[129][2]                   
                                                                       lambda_41[0][0]                  
                                                                       lstm_1[130][0]                   
                                                                       lstm_1[130][2]                   
                                                                       lambda_42[0][0]                  
                                                                       lstm_1[131][0]                   
                                                                       lstm_1[131][2]                   
                                                                       lambda_43[0][0]                  
                                                                       lstm_1[132][0]                   
                                                                       lstm_1[132][2]                   
                                                                       lambda_44[0][0]                  
                                                                       lstm_1[133][0]                   
                                                                       lstm_1[133][2]                   
                                                                       lambda_45[0][0]                  
                                                                       lstm_1[134][0]                   
                                                                       lstm_1[134][2]                   
                                                                       lambda_46[0][0]                  
                                                                       lstm_1[135][0]                   
                                                                       lstm_1[135][2]                   
                                                                       lambda_47[0][0]                  
                                                                       lstm_1[136][0]                   
                                                                       lstm_1[136][2]                   
                                                                       lambda_48[0][0]                  
                                                                       lstm_1[137][0]                   
                                                                       lstm_1[137][2]                   
                                                                       lambda_49[0][0]                  
                                                                       lstm_1[138][0]                   
                                                                       lstm_1[138][2]                   
                                                                       lambda_50[0][0]                  
                                                                       lstm_1[139][0]                   
                                                                       lstm_1[139][2]                   
                                                                       lambda_51[0][0]                  
                                                                       lstm_1[140][0]                   
                                                                       lstm_1[140][2]                   
                                                                       lambda_52[0][0]                  
                                                                       lstm_1[141][0]                   
                                                                       lstm_1[141][2]                   
                                                                       lambda_53[0][0]                  
                                                                       lstm_1[142][0]                   
                                                                       lstm_1[142][2]                   
                                                                       lambda_54[0][0]                  
                                                                       lstm_1[143][0]                   
                                                                       lstm_1[143][2]                   
                                                                       lambda_55[0][0]                  
                                                                       lstm_1[144][0]                   
                                                                       lstm_1[144][2]                   
                                                                       lambda_56[0][0]                  
                                                                       lstm_1[145][0]                   
                                                                       lstm_1[145][2]                   
                                                                       lambda_57[0][0]                  
                                                                       lstm_1[146][0]                   
                                                                       lstm_1[146][2]                   
                                                                       lambda_58[0][0]                  
                                                                       lstm_1[147][0]                   
                                                                       lstm_1[147][2]                   
                                                                       lambda_59[0][0]                  
                                                                       lstm_1[148][0]                   
                                                                       lstm_1[148][2]                   
                                                                       lambda_60[0][0]                  
                                                                       lstm_1[149][0]                   
                                                                       lstm_1[149][2]                   
                                                                       lambda_61[0][0]                  
                                                                       lstm_1[150][0]                   
                                                                       lstm_1[150][2]                   
                                                                       lambda_62[0][0]                  
                                                                       lstm_1[151][0]                   
                                                                       lstm_1[151][2]                   
                                                                       lambda_63[0][0]                  
                                                                       lstm_1[152][0]                   
                                                                       lstm_1[152][2]                   
                                                                       lambda_64[0][0]                  
                                                                       lstm_1[153][0]                   
                                                                       lstm_1[153][2]                   
                                                                       lambda_65[0][0]                  
                                                                       lstm_1[154][0]                   
                                                                       lstm_1[154][2]                   
                                                                       lambda_66[0][0]                  
                                                                       lstm_1[155][0]                   
                                                                       lstm_1[155][2]                   
                                                                       lambda_67[0][0]                  
                                                                       lstm_1[156][0]                   
                                                                       lstm_1[156][2]                   
                                                                       lambda_68[0][0]                  
                                                                       lstm_1[157][0]                   
                                                                       lstm_1[157][2]                   
                                                                       lambda_69[0][0]                  
                                                                       lstm_1[158][0]                   
                                                                       lstm_1[158][2]                   
                                                                       lambda_70[0][0]                  
                                                                       lstm_1[159][0]                   
                                                                       lstm_1[159][2]                   
                                                                       lambda_71[0][0]                  
                                                                       lstm_1[160][0]                   
                                                                       lstm_1[160][2]                   
                                                                       lambda_72[0][0]                  
                                                                       lstm_1[161][0]                   
                                                                       lstm_1[161][2]                   
                                                                       lambda_73[0][0]                  
                                                                       lstm_1[162][0]                   
                                                                       lstm_1[162][2]                   
                                                                       lambda_74[0][0]                  
                                                                       lstm_1[163][0]                   
                                                                       lstm_1[163][2]                   
                                                                       lambda_75[0][0]                  
                                                                       lstm_1[164][0]                   
                                                                       lstm_1[164][2]                   
                                                                       lambda_76[0][0]                  
                                                                       lstm_1[165][0]                   
                                                                       lstm_1[165][2]                   
                                                                       lambda_77[0][0]                  
                                                                       lstm_1[166][0]                   
                                                                       lstm_1[166][2]                   
                                                                       lambda_78[0][0]                  
                                                                       lstm_1[167][0]                   
                                                                       lstm_1[167][2]                   
                                                                       lambda_79[0][0]                  
                                                                       lstm_1[168][0]                   
                                                                       lstm_1[168][2]                   
                                                                       lambda_80[0][0]                  
                                                                       lstm_1[169][0]                   
                                                                       lstm_1[169][2]                   
                                                                       lambda_81[0][0]                  
                                                                       lstm_1[170][0]                   
                                                                       lstm_1[170][2]                   
    ____________________________________________________________________________________________________
    dense_1 (Dense)                  (None, 78)            5070        lstm_1[122][0]                   
                                                                       lstm_1[123][0]                   
                                                                       lstm_1[124][0]                   
                                                                       lstm_1[125][0]                   
                                                                       lstm_1[126][0]                   
                                                                       lstm_1[127][0]                   
                                                                       lstm_1[128][0]                   
                                                                       lstm_1[129][0]                   
                                                                       lstm_1[130][0]                   
                                                                       lstm_1[131][0]                   
                                                                       lstm_1[132][0]                   
                                                                       lstm_1[133][0]                   
                                                                       lstm_1[134][0]                   
                                                                       lstm_1[135][0]                   
                                                                       lstm_1[136][0]                   
                                                                       lstm_1[137][0]                   
                                                                       lstm_1[138][0]                   
                                                                       lstm_1[139][0]                   
                                                                       lstm_1[140][0]                   
                                                                       lstm_1[141][0]                   
                                                                       lstm_1[142][0]                   
                                                                       lstm_1[143][0]                   
                                                                       lstm_1[144][0]                   
                                                                       lstm_1[145][0]                   
                                                                       lstm_1[146][0]                   
                                                                       lstm_1[147][0]                   
                                                                       lstm_1[148][0]                   
                                                                       lstm_1[149][0]                   
                                                                       lstm_1[150][0]                   
                                                                       lstm_1[151][0]                   
                                                                       lstm_1[152][0]                   
                                                                       lstm_1[153][0]                   
                                                                       lstm_1[154][0]                   
                                                                       lstm_1[155][0]                   
                                                                       lstm_1[156][0]                   
                                                                       lstm_1[157][0]                   
                                                                       lstm_1[158][0]                   
                                                                       lstm_1[159][0]                   
                                                                       lstm_1[160][0]                   
                                                                       lstm_1[161][0]                   
                                                                       lstm_1[162][0]                   
                                                                       lstm_1[163][0]                   
                                                                       lstm_1[164][0]                   
                                                                       lstm_1[165][0]                   
                                                                       lstm_1[166][0]                   
                                                                       lstm_1[167][0]                   
                                                                       lstm_1[168][0]                   
                                                                       lstm_1[169][0]                   
                                                                       lstm_1[170][0]                   
                                                                       lstm_1[171][0]                   
    ____________________________________________________________________________________________________
    lambda_33 (Lambda)               (None, 1, 78)         0           dense_1[122][0]                  
    ____________________________________________________________________________________________________
    lambda_34 (Lambda)               (None, 1, 78)         0           dense_1[123][0]                  
    ____________________________________________________________________________________________________
    lambda_35 (Lambda)               (None, 1, 78)         0           dense_1[124][0]                  
    ____________________________________________________________________________________________________
    lambda_36 (Lambda)               (None, 1, 78)         0           dense_1[125][0]                  
    ____________________________________________________________________________________________________
    lambda_37 (Lambda)               (None, 1, 78)         0           dense_1[126][0]                  
    ____________________________________________________________________________________________________
    lambda_38 (Lambda)               (None, 1, 78)         0           dense_1[127][0]                  
    ____________________________________________________________________________________________________
    lambda_39 (Lambda)               (None, 1, 78)         0           dense_1[128][0]                  
    ____________________________________________________________________________________________________
    lambda_40 (Lambda)               (None, 1, 78)         0           dense_1[129][0]                  
    ____________________________________________________________________________________________________
    lambda_41 (Lambda)               (None, 1, 78)         0           dense_1[130][0]                  
    ____________________________________________________________________________________________________
    lambda_42 (Lambda)               (None, 1, 78)         0           dense_1[131][0]                  
    ____________________________________________________________________________________________________
    lambda_43 (Lambda)               (None, 1, 78)         0           dense_1[132][0]                  
    ____________________________________________________________________________________________________
    lambda_44 (Lambda)               (None, 1, 78)         0           dense_1[133][0]                  
    ____________________________________________________________________________________________________
    lambda_45 (Lambda)               (None, 1, 78)         0           dense_1[134][0]                  
    ____________________________________________________________________________________________________
    lambda_46 (Lambda)               (None, 1, 78)         0           dense_1[135][0]                  
    ____________________________________________________________________________________________________
    lambda_47 (Lambda)               (None, 1, 78)         0           dense_1[136][0]                  
    ____________________________________________________________________________________________________
    lambda_48 (Lambda)               (None, 1, 78)         0           dense_1[137][0]                  
    ____________________________________________________________________________________________________
    lambda_49 (Lambda)               (None, 1, 78)         0           dense_1[138][0]                  
    ____________________________________________________________________________________________________
    lambda_50 (Lambda)               (None, 1, 78)         0           dense_1[139][0]                  
    ____________________________________________________________________________________________________
    lambda_51 (Lambda)               (None, 1, 78)         0           dense_1[140][0]                  
    ____________________________________________________________________________________________________
    lambda_52 (Lambda)               (None, 1, 78)         0           dense_1[141][0]                  
    ____________________________________________________________________________________________________
    lambda_53 (Lambda)               (None, 1, 78)         0           dense_1[142][0]                  
    ____________________________________________________________________________________________________
    lambda_54 (Lambda)               (None, 1, 78)         0           dense_1[143][0]                  
    ____________________________________________________________________________________________________
    lambda_55 (Lambda)               (None, 1, 78)         0           dense_1[144][0]                  
    ____________________________________________________________________________________________________
    lambda_56 (Lambda)               (None, 1, 78)         0           dense_1[145][0]                  
    ____________________________________________________________________________________________________
    lambda_57 (Lambda)               (None, 1, 78)         0           dense_1[146][0]                  
    ____________________________________________________________________________________________________
    lambda_58 (Lambda)               (None, 1, 78)         0           dense_1[147][0]                  
    ____________________________________________________________________________________________________
    lambda_59 (Lambda)               (None, 1, 78)         0           dense_1[148][0]                  
    ____________________________________________________________________________________________________
    lambda_60 (Lambda)               (None, 1, 78)         0           dense_1[149][0]                  
    ____________________________________________________________________________________________________
    lambda_61 (Lambda)               (None, 1, 78)         0           dense_1[150][0]                  
    ____________________________________________________________________________________________________
    lambda_62 (Lambda)               (None, 1, 78)         0           dense_1[151][0]                  
    ____________________________________________________________________________________________________
    lambda_63 (Lambda)               (None, 1, 78)         0           dense_1[152][0]                  
    ____________________________________________________________________________________________________
    lambda_64 (Lambda)               (None, 1, 78)         0           dense_1[153][0]                  
    ____________________________________________________________________________________________________
    lambda_65 (Lambda)               (None, 1, 78)         0           dense_1[154][0]                  
    ____________________________________________________________________________________________________
    lambda_66 (Lambda)               (None, 1, 78)         0           dense_1[155][0]                  
    ____________________________________________________________________________________________________
    lambda_67 (Lambda)               (None, 1, 78)         0           dense_1[156][0]                  
    ____________________________________________________________________________________________________
    lambda_68 (Lambda)               (None, 1, 78)         0           dense_1[157][0]                  
    ____________________________________________________________________________________________________
    lambda_69 (Lambda)               (None, 1, 78)         0           dense_1[158][0]                  
    ____________________________________________________________________________________________________
    lambda_70 (Lambda)               (None, 1, 78)         0           dense_1[159][0]                  
    ____________________________________________________________________________________________________
    lambda_71 (Lambda)               (None, 1, 78)         0           dense_1[160][0]                  
    ____________________________________________________________________________________________________
    lambda_72 (Lambda)               (None, 1, 78)         0           dense_1[161][0]                  
    ____________________________________________________________________________________________________
    lambda_73 (Lambda)               (None, 1, 78)         0           dense_1[162][0]                  
    ____________________________________________________________________________________________________
    lambda_74 (Lambda)               (None, 1, 78)         0           dense_1[163][0]                  
    ____________________________________________________________________________________________________
    lambda_75 (Lambda)               (None, 1, 78)         0           dense_1[164][0]                  
    ____________________________________________________________________________________________________
    lambda_76 (Lambda)               (None, 1, 78)         0           dense_1[165][0]                  
    ____________________________________________________________________________________________________
    lambda_77 (Lambda)               (None, 1, 78)         0           dense_1[166][0]                  
    ____________________________________________________________________________________________________
    lambda_78 (Lambda)               (None, 1, 78)         0           dense_1[167][0]                  
    ____________________________________________________________________________________________________
    lambda_79 (Lambda)               (None, 1, 78)         0           dense_1[168][0]                  
    ____________________________________________________________________________________________________
    lambda_80 (Lambda)               (None, 1, 78)         0           dense_1[169][0]                  
    ____________________________________________________________________________________________________
    lambda_81 (Lambda)               (None, 1, 78)         0           dense_1[170][0]                  
    ====================================================================================================
    Total params: 41,678
    Trainable params: 41,678
    Non-trainable params: 0
    ____________________________________________________________________________________________________


** Expected Output**
If you scroll to the bottom of the output, you'll see:
```
Total params: 41,678
Trainable params: 41,678
Non-trainable params: 0
```

#### Initialize inference model
The following code creates the zero-valued vectors you will use to initialize `x` and the LSTM state variables `a` and `c`. 


```python
x_initializer = np.zeros((1, 1, 78))
a_initializer = np.zeros((1, n_a))
c_initializer = np.zeros((1, n_a))
```

**Exercise**: Implement `predict_and_sample()`. 

* This function takes many arguments including the inputs [x_initializer, a_initializer, c_initializer]. 
* In order to predict the output corresponding to this input, you will need to carry-out 3 steps:


#### Step 1
* Use your inference model to predict an output given your set of inputs. The output `pred` should be a list of length $T_y$ where each element is a numpy-array of shape (1, n_values).
```Python
inference_model.predict([input_x_init, hidden_state_init, cell_state_init])
```
    * Choose the appropriate input arguments to `predict` from the input arguments of this `predict_and_sample` function.

#### Step 2
* Convert `pred` into a numpy array of $T_y$ indices. 
    * Each index is computed by taking the `argmax` of an element of the `pred` list. 
    * Use [numpy.argmax](https://docs.scipy.org/doc/numpy/reference/generated/numpy.argmax.html).
    * Set the `axis` parameter.
        * Remember that the shape of the prediction is $(m, T_{y}, n_{values})$

#### Step 3  
* Convert the indices into their one-hot vector representations. 
    * Use [to_categorical](https://keras.io/utils/#to_categorical).
    * Set the `num_classes` parameter. Note that for grading purposes: you'll need to either:
        * Use a dimension from the given parameters of `predict_and_sample()` (for example, one of the dimensions of x_initializer has the value for the number of distinct classes).
        * Or just hard code the number of distinct classes (will pass the grader as well).
        * Note that using a global variable such as n_values will not work for grading purposes.


```python
# GRADED FUNCTION: predict_and_sample

def predict_and_sample(inference_model, x_initializer = x_initializer, a_initializer = a_initializer, 
                       c_initializer = c_initializer):
    """
    Predicts the next value of values using the inference model.
    
    Arguments:
    inference_model -- Keras model instance for inference time
    x_initializer -- numpy array of shape (1, 1, 78), one-hot vector initializing the values generation
    a_initializer -- numpy array of shape (1, n_a), initializing the hidden state of the LSTM_cell
    c_initializer -- numpy array of shape (1, n_a), initializing the cell state of the LSTM_cel
    
    Returns:
    results -- numpy-array of shape (Ty, 78), matrix of one-hot vectors representing the values generated
    indices -- numpy-array of shape (Ty, 1), matrix of indices representing the values generated
    """
    
    ### START CODE HERE ###
    # Step 1: Use your inference model to predict an output sequence given x_initializer, a_initializer and c_initializer.
    pred = inference_model.predict([x_initializer, a_initializer, c_initializer])
    # Step 2: Convert "pred" into an np.array() of indices with the maximum probabilities
    indices = np.argmax(pred,axis=-1)
    # Step 3: Convert indices to one-hot vectors, the shape of the results should be (Ty, n_values)
    results = to_categorical(indices,num_classes=x_initializer.shape[-1])
    ### END CODE HERE ###
    
    return results, indices
```


```python
results, indices = predict_and_sample(inference_model, x_initializer, a_initializer, c_initializer)
print("np.argmax(results[12]) =", np.argmax(results[12]))
print("np.argmax(results[17]) =", np.argmax(results[17]))
print("list(indices[12:18]) =", list(indices[12:18]))
```

    np.argmax(results[12]) = 56
    np.argmax(results[17]) = 12
    list(indices[12:18]) = [array([56]), array([12]), array([11]), array([15]), array([56]), array([12])]


**Expected (Approximate) Output**: 

* Your results **may likely differ** because Keras' results are not completely predictable. 
* However, if you have trained your LSTM_cell with model.fit() for exactly 100 epochs as described above: 
    * You should very likely observe a sequence of indices that are not all identical. 
    * Moreover, you should observe that: 
        * np.argmax(results[12]) is the first element of list(indices[12:18]) 
        * and np.argmax(results[17]) is the last element of list(indices[12:18]). 

<table>
    <tr>
        <td>
            **np.argmax(results[12])** =
        </td>
        <td>
        1
        </td>
    </tr>
    <tr>
        <td>
            **np.argmax(results[17])** =
        </td>
        <td>
        42
        </td>
    </tr>
    <tr>
        <td>
            **list(indices[12:18])** =
        </td>
        <td>
            [array([1]), array([42]), array([54]), array([17]), array([1]), array([42])]
        </td>
    </tr>
</table>

#### 3.3 - Generate music 

Finally, you are ready to generate music. Your RNN generates a sequence of values. The following code generates music by first calling your `predict_and_sample()` function. These values are then post-processed into musical chords (meaning that multiple values or notes can be played at the same time). 

Most computational music algorithms use some post-processing because it is difficult to generate music that sounds good without such post-processing. The post-processing does things such as clean up the generated audio by making sure the same sound is not repeated too many times, that two successive notes are not too far from each other in pitch, and so on. One could argue that a lot of these post-processing steps are hacks; also, a lot of the music generation literature has also focused on hand-crafting post-processors, and a lot of the output quality depends on the quality of the post-processing and not just the quality of the RNN. But this post-processing does make a huge difference, so let's use it in our implementation as well. 

Let's make some music! 

Run the following cell to generate music and record it into your `out_stream`. This can take a couple of minutes.


```python
out_stream = generate_music(inference_model)
```

    Predicting new values for different set of chords.
    Generated 51 sounds using the predicted values for the set of chords ("1") and after pruning
    Generated 51 sounds using the predicted values for the set of chords ("2") and after pruning
    Generated 51 sounds using the predicted values for the set of chords ("3") and after pruning
    Generated 50 sounds using the predicted values for the set of chords ("4") and after pruning
    Generated 51 sounds using the predicted values for the set of chords ("5") and after pruning
    Your generated music is saved in output/my_music.midi


To listen to your music, click File->Open... Then go to "output/" and download "my_music.midi". Either play it on your computer with an application that can read midi files if you have one, or use one of the free online "MIDI to mp3" conversion tools to convert this to mp3.  

As a reference, here is a 30 second audio clip we generated using this algorithm. 


```python
IPython.display.Audio('./data/30s_trained_model.mp3')
```





    




### Congratulations!

You have come to the end of the notebook. 


## What you should remember
- A sequence model can be used to generate musical values, which are then post-processed into midi music. 
- Fairly similar models can be used to generate dinosaur names or to generate music, with the major difference being the input fed to the model.  
- In Keras, sequence generation involves defining layers with shared weights, which are then repeated for the different time steps $1, \ldots, T_x$. 

Congratulations on completing this assignment and generating a jazz solo! 

**References**

The ideas presented in this notebook came primarily from three computational music papers cited below. The implementation here also took significant inspiration and used many components from Ji-Sung Kim's GitHub repository.

- Ji-Sung Kim, 2016, [deepjazz](https://github.com/jisungk/deepjazz)
- Jon Gillick, Kevin Tang and Robert Keller, 2009. [Learning Jazz Grammars](http://ai.stanford.edu/~kdtang/papers/smc09-jazzgrammar.pdf)
- Robert Keller and David Morrison, 2007, [A Grammatical Approach to Automatic Improvisation](http://smc07.uoa.gr/SMC07%20Proceedings/SMC07%20Paper%2055.pdf)
- François Pachet, 1999, [Surprising Harmonies](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.5.7473&rep=rep1&type=pdf)

We're also grateful to François Germain for valuable feedback.


```python

```
