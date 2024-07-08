## 暂定

- BigDataModel.Model
  - iTransformer
    - 输入->lstm(last hidden) 时序嵌入
    - hidden按时序维嵌入
    - iTransformer其余部分
  - fourier attention
    - feature embedding
    - rfft — feature attention – irfft
    - lstm 递归预测

## TODO

- Model 主体

  - LSTM相关
  - iTransformer，LSTM输入
  - FAt，LSTM输出
  - 带参数的指数平滑后处理
- 学习率调度
- 其他时序attention
