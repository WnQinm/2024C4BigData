## 暂定

- Baseline
  - wind mse 4.4
  - temp mse 10.2

- Pyraformer
  - wind mse 3.62


## TODO

- 前期用大batch训练以快速下降loss
  - baseline（参数量几十k）用24g的3090bs40960num_worker5（占21g左右）半个小时一轮
  - Pyraformer（参数量几百k）用24g3090bs10240num_worker5（占19g左右）一个半小时一轮
    - 测多卡3090
    - 测48gL20bs20480
    - 测80gA800bs40960
- ~~BigDataModel.Model~~
  - ~~iTransformer~~
  - fourier attention
    - ~~rfft - attention - irfft~~
  - lstm 递归预测
  - ~~指数平滑~~
- ~~学习率调度~~
- 其他时序attention
