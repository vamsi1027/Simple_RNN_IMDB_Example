import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

sample_text = """Enter Claudius King of Denmarke, Gertrude the Queene, Hamlet,
Polonius,
Laertes, and his Sister Ophelia, Lords Attendant.

King. Though yet of Hamlet our deere Brothers death
The memory be greene: and that it vs befitted
To beare our hearts in greefe, and our whole Kingdome
To be contracted in one brow of woe:
Yet so farre hath Discretion fought with Nature,
That we with wisest sorrow thinke on him,
Together with remembrance of our selues.
Therefore our sometimes Sister, now our Queene,
Th' imperiall Ioyntresse of this warlike State,
Haue we, as 'twere, with a defeated ioy,
With one Auspicious, and one Dropping eye,
With mirth in Funerall, and with Dirge in Marriage,
In equall Scale weighing Delight and Dole
Taken to Wife; nor haue we heerein barr'd
Your better Wisedomes, which haue freely gone
With this affaire along, for all our Thankes.
Now followes, that you know young Fortinbras,
Holding a weake supposall of our worth;
Or thinking by our late deere Brothers death,
Our State to be disioynt, and out of Frame,
Colleagued with the dreame of his Aduantage;
He hath not fayl'd to pester vs with Message,
Importing the surrender of those Lands
Lost by his Father: with all Bonds of Law
To our most valiant Brother. So much for him.
Enter Voltemand and Cornelius.
"""

## Load the LSTM model
model = load_model('next_word_prediction_model.h5')


## Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

## Function to predict the next word in a sequence
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len :
        token_list = token_list[-(max_sequence_len-1):] # Ensure the sequence length matches max_sequence_len - 1
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=-1)[0]
    
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

## streamlit app
st.title("Next word prediction with lstm and early stopping")
st.write(sample_text)
input_text = st.text_input("enter a sequence of words:", "to be or not to be")
if st.button("predict next word"):
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len=10)
    st.write(f"predicted next word: {next_word}")


