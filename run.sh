## A:[0,1], B:[3], C:[1,3], D:[1]
#python main.py --branches=[[0,1]]  > ../txts/chapter5_1_2/1.txt 2>&1
#python main.py --branches=[[3]]  > ../txts/chapter5_1_2/2.txt 2>&1
#python main.py --branches=[[1,3]]  > ../txts/chapter5_1_2/3.txt 2>&1
#python main.py --branches=[[1]]  > ../txts/chapter5_1_2/4.txt 2>&1
#python main.py --branches=[[0,1],[3]]  > ../txts/chapter5_1_2/5.txt 2>&1
#python main.py --branches=[[0,1],[1,3]]  > ../txts/chapter5_1_2/6.txt 2>&1
#python main.py --branches=[[0,1],[1]]  > ../txts/chapter5_1_2/7.txt 2>&1
#python main.py --branches=[[3],[1,3]]  > ../txts/chapter5_1_2/8.txt 2>&1
#python main.py --branches=[[3],[1]]  > ../txts/chapter5_1_2/9.txt 2>&1
#python main.py --branches=[[1,3],[1]]  > ../txts/chapter5_1_2/10.txt 2>&1
#python main.py --branches=[[0,1],[3],[1,3]]  > ../txts/chapter5_1_2/11.txt 2>&1
#python main.py --branches=[[0,1],[3],[1]]  > ../txts/chapter5_1_2/12.txt 2>&1
#python main.py --branches=[[0,1],[1,3],[1]]  > ../txts/chapter5_1_2/13.txt 2>&1
#python main.py --branches=[[3],[1,3],[1]]  > ../txts/chapter5_1_2/14.txt 2>&1
#python main.py --branches=[[0,1],[3],[1,3],[1]]  > ../txts/chapter5_1_2/15.txt 2>&1
#python main.py  > ../txts/chapter5_1_2/0.txt 2>&1

python main.py --branches=[[0,1],[3],[1,3],[1]] --fold=1 > ../txts/chapter511_and_52_53/115.txt 2>&1
python main.py --branches=[[0,1],[3],[1,3],[1]] --fold=2 > ../txts/chapter511_and_52_53/116.txt 2>&1
python main.py --branches=[[0,1],[3],[1,3],[1]] --fold=3 > ../txts/chapter511_and_52_53/117.txt 2>&1
python main.py --branches=[[0,1],[3],[1,3],[1]] --fold=4 > ../txts/chapter511_and_52_53/118.txt 2>&1
python main.py --branches=[[0,1],[3],[1,3],[1]] --fold=5 > ../txts/chapter511_and_52_53/119.txt 2>&1