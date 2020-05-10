# MeanReversionNeuralNetowrkPredictingStockIndex
An example of processing input data from a stock index and of training Neural Network to predict the direction of the index. Additionally, this repo shows some application to AWS cloud and iOS App.

■ Abstruct ■
  ... making ...

■ Contents ■
  1. Data Proccesing
  2. Custamize tha input data and Machine Larning
  3. Prediction and Interpretation
  4. AWS and iOS Application

■ 1. Data Proccesing

  First of all, we need the input data. 
  The file "ratio.py" generates the input for the neural network from the raw stock prices.

   To start with, The code finds maximum poles and minimum poles on the moving average(curve). From them, it decides "peaks and bottoms" of the stock price during the terms.
   Then, it generates accelarations(UP_RATIOs) from each peak and bottom. The accelarations mean the speeds of differences from the peaks and bottoms.
   
   The input data for the neural network consits of 25 days' differences of stock prices from moving averages and the accelaration at the day as the example(ex_input.csv) shows.
   
! Disclaimer: The output(the input for neural network) possibly includes exceptions.
              Because of the "peaks and bottoms" are difined by artificiallly, mechanically, or judged by the algolism bellow shows for proccesing such a  big data in a short time, cases are that, there possibly includes some exceptions, which doesn't seems to be the peaks or bottoms in the term.



... making ...
■ 2.
■ 3.
■ 4.
