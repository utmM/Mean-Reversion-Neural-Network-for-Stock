# MeanReversionNeuralNetowrkPredictingStockIndex

"Mean Reversion", or Moving Average Analysis is the most popular way of predicting stock prices.
Things retruns to its average.

Using the simple structure of the neural network, the divergence rates sequence from moving average curve gives a prediction of up/down direction about stock price.

The neural network structure modeled by tensorfllow is simple and audinally one.

As is usual, data pre-processing or preparetion play the most important part on machine learning.
This repo focuses on showing an example of processing input data from row stock price and of an interpretation about the output.

Also explains about predicting by the Simple Neural Network.
Additionally, shows some applications to develop an iOS App through the AWS cloud.

■ Contents ■

  1. Data Proccesing
  2. Enphasis the meaning of inputs and Machine Learning
  3. Prediction and output interpretation
  4. AWS and iOS Application

■ 1. Data Proccesing

  First, we develop the input data. The file "ratio.py" generates the input for the neural network.

  (Command)
  A. Move the foloder (using -cd command etc.) including ratio.py and row data. (Raw data example is 
  in the repo named ex_N225_index.csv)
  B. Run by -python command. (On terminal, "python ratio.py")

  (Explanation)
   Algo in "ratio.py" finds maximum poles and minimum poles on the moving average(curve). From them, it decides "peaks and bottoms" of the price during the terms. Then, it generates accelarations (UP_RATIOs) from each peak and bottom. The accelaration means the speeds of differences from the peaks and bottoms. The input data for the neural network consits of 25 days' differences of stock prices from moving average and the accelaration at the day, as the example (ex_input.csv) shows.
   
  (! Disclaimer)
   The output(the input for neural network) possibly includes exceptions. Because of the "peaks and bottoms" are difined by artificiallly, mechanically, or judged by the algolism bellow shows for proccesing such a  big data in a short time, cases are that, there possibly includes some exceptions, which doesn't seems to be the peaks or bottoms in the term.

■ 2.Emphasis the meaning of inputs and Machine Learning

 Interpretation of both input and output data is essencial for the simple NN. 
 To make sure about the NN can understand the meaning of a data sequence, some data custamization is needed.
 Enphasis or leveling before the machine learning.
 
 (Emphasize)
 
 
 
 (Command)
 A. -cd to the folder including (nn_stock.py, input.csv)
 B. python nn_stock.py
 
 
 (Machine Learning)
  Trained by the 80% of the input data as teacher data (rest 20% are for the test data), the simple neural network gives a prediction of stock direction (acceralation).
   25 days' differences of current prices as the input for the NN gives 1 predicted accelaration.
   
   Tuning and adjustment of the range which is the correct answer or not for prediction is important.
   (This is up to users)

■ 3. ... making ...
■ 4. ... making ...

