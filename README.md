# 隐含属性理解（人脑子系统）
使用与子系统有关的辅助任务，结果见[这里](https://q9eurnjld4.feishu.cn/sheets/shtcnEznwEEb1zWrvWXdsnWFYqh)  
## 运行示例及run_sh说明  
A:[0,1], B:[3], C:[1,3], D:[1]  
A，B，C，D即论文中的A，B，C，D分类器 
branches里可以写ABCD的任意组合。
~~~shell
python main.py --branches=[[0,1]]  # A  
python main.py --branches=[[3]]  # B  
python main.py --branches=[[1,3]]  # C  
python main.py --branches=[[1]]  # D  
python main.py --branches=[[0,1],[3]]  # A+B  
python main.py --branches=[[0,1],[1,3]]  # AC  
python main.py --branches=[[0,1],[1]]  # AD  
python main.py --branches=[[3],[1,3]]  # BC  
python main.py --branches=[[3],[1]]  # BD  
python main.py --branches=[[1,3],[1]]  # CD   
python main.py --branches=[[0,1],[3],[1,3]]  # ABC  
python main.py --branches=[[0,1],[3],[1]]  # ABD  
python main.py --branches=[[0,1],[1,3],[1]]  # ACD  
python main.py --branches=[[3],[1,3],[1]]  # BCD  
python main.py --branches=[[0,1],[3],[1,3],[1]]  # ABCD   
~~~
### 交叉验证  
~~~shell
python main.py --branches=[[0,1],[3],[1,3],[1]] --fold=1
python main.py --branches=[[0,1],[3],[1,3],[1]] --fold=2
python main.py --branches=[[0,1],[3],[1,3],[1]] --fold=3
python main.py --branches=[[0,1],[3],[1,3],[1]] --fold=4
python main.py --branches=[[0,1],[3],[1,3],[1]] --fold=5
~~~
# 注意  
和PAD一样，涉及多任务的部分，test loss仅计算了主任务的loss以不影响测试时forward
###以下忽略
#### update: 调整使用最佳时序处理  
基于时序信息处理的最佳模型：GRU+drop low attention[here](https://gitlab.com/bupt_pris_ser/baseline/-/blob/chapter5.2.2/model.py)  
由于GPU原因，去另一台的3090上跑。结果不好，因此不使用最佳时序处理。  
#### again update： 使用5.3策略  
还是另一台的3090
