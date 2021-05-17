# AI_homework3

---------------------使用說明----------------------

train.py為訓練程式碼

若要執行請將194行的路徑指向自己的target路徑，並且target0~49都要在同個資料夾

必須從""原始碼執行""

main.py為預測程式碼

若要執行請以以下指令執行

python main.py --consumption "路徑" --generation "路徑"  --bidresult "路徑" --output "路徑"

或者用pipenv創建環境執行

---------------------網路架構----------------------

我所使用的模型為 LSTM

以下為我的LSTM模型參數

     input_size=2,
     
     hidden_size=128,
     
     num_layers=2,
     

並且我的LSTM的輸出是接第二層layer最後一個step的輸出

然後接上一個全連接進行最終輸出

以下為全連接參數

    fc=nn.Linear(128,64)  //全連接第一層
    
    relu=nn.ReLU(inplace=True) // activation function
    
    fc_2=nn.Linear(64,2)   //全連接第二層
    
        
---------------------訓練方式----------------------

training:

  每次用前168小時去預測第169小時
  
訓練參數:

     batch_size=32

     loss function=RMSE
  
Data Normalization:

     generation_mean=0.7808

     Consumption_mean=1.4447

     generation_std=1.1422

     Consumption_std=1.1166


  
---------------------預測手法----------------------

  用1~168小時預測第169小時
  
  在用2~168再加上預測的169，去預測170
  
  重複循環至跑完24小時

----------------------買賣決策----------------------

如果預測出來的小時 Generation 大於 Consumption 則 賣

以random.randrange(100, 200, 1)/100 來出價

如果預測出來的小時 Generation 小於 Consumption 則 買

以random.randrange(50, 150, 1)/100)

