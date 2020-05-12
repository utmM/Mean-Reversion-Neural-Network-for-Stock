# MeanReversionNeuralNetowrkForStock

"Mean Reversion", or Moving Average Analysis is the most popular way of predicting stock prices.
Things retruns to its average.

Using the simple structure of the neural network, the divergence rates sequence from moving average curve gives a prediction of up/down direction about stock price.

The neural network structure modeled by tensorfllow is simple and audinally one.

As is usual, data pre-processing or preparation play the most important part on machine learning.
This repo focuses on showing an example of processing input data from row stock price and of an interpretation about the output.

Also explains about predicting by the Simple Neural Network.
Additionally, shows some applications to develop an iOS App through the AWS cloud.

■ Contents ■

  1. Data Proccesing
  2. Emphasize meaning of inputs and Machine Learning
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

■ 2.Emphasize the meaning of inputs and Machine Learning

 Interpretation of both input and output data is essencial for the simple NN. 
 
 (★Emphasize)
 Before running this code, you need to adjust of the input data to Emphasize the meaning of the time series.
    "UP_RATIO (Accelaration)" is, to be adjusted in the 2 steps bellow.
    
    Step 1.
    Adopt the accelaration which gives the maximum absolete during past 5 days.
    
    Step 2.
    On Step 1. accelaratons, adopt the one which has the maximum absolete during 5 days, from 2 days before to 2 days past, including the day.
    
    Step3.
    Normalize the accelarations by its maximum and minimum during all the term.
    
 The NN uses the adjusted accelarations together with 25 differencials as input.
    Please renew the "input.csv" by adjudted UP_RATIO (accelaration)
    -> A line consists of 25 differencials and adjusted accelaration.
    Renamed the input as ./input.csv

 (Why need the adjustments?)
    The row accelaration data has large variation or deviation so the NN can't read the meaning of the time series. Before input, emphasize the trend, make it easy to understand for the NN.
 
 (Machine Learning)
  Trained by the 80% of the input data as teacher data (rest 20% are for the test data), the simple neural network gives a prediction of stock direction (acceralation).
   25 days' differences of current prices as the input for the NN gives 1 predicted accelaration.
   
   Tuning and adjustment of the range which is the correct answer or not for prediction is important.
   (This is up to the users)
   
   (Command)
 After adjusting accelaration (:Step 1~3) and setting of the error range, enter the command:
 A. -cd to the folder including (nn_stock.py, input.csv)
 B. python nn_stock.py

■ 3. Prediction and output interpretation
■ 4. AWS and iOS Application

