# 离散注水算法
import winsound
import datetime
import numpy as np
import scipy.special
from multiprocessing import Pool
from matplotlib import pyplot as plt
np.set_printoptions(precision = 8)       # 浮点数组输出的精度位数，即小数点后位数(默认为8)。
np.set_printoptions(threshold = 1000000) # 元素门槛值。数组个数沒有超过设置的阈值，NumPy就会将Python將所有元素列印出來。

# 最小化凸函数，即对凹函数取负
# np.seterr(divide='ignore',invalid='ignore')  忽略所有警告
# e是障碍法中的参数1/t，c为回溯直线搜索中α，β取为0.5
# Ynm的累计分布函数服从指数分布
N = 40        # N个汽车，M个服务器，
M = 10
Cnm = np.zeros((N, M))                   # 单个副本的成本，默认为1
Cn = np.zeros((N,1))                     # 每个子任务的总成本限制
Cn2 = np.zeros((N,1))                    # 当前资源剩余预算
x = np.zeros((N*M,1))
obj = []                                 # 存放每一次迭代后任务中断概率的取值
Pm = np.zeros((M,1))                                         # 服务器m最大能并行执行的任务副本数
Pm2 = np.zeros((M,1))                                        # 当前剩余资源量


# 随机分布
np.random.seed(1)                                           # 使用相同的seed()值，保证同一程序每次运行结果一样
# r = np.random.choice(N*M,size=N*M,replace=False,p=None)   # 在[0,N*M)中随机选取
r = np.random.randint(50,100,size=(1, N*M))/100
Fx = np.array(r).reshape((1,N*M))                           # Ynm的分布函数，Pr(Ynm<=dt)的取值
print("随机数r为:",r)
print("Pr(Ynm<=dt)取值:", Fx)
print("Pr(Ynm<=dt)增序取值:", np.sort(Fx, axis=None))       # 随机分布



# 初始化使所有任务至少一个副本
def init():
    np.random.seed(1)
    temp, index = 0, 0
    for i in range(N):
        Cn[i, 0] = np.random.randint(5, 20)  # 随机初始化每个子任务n的成本
    np.random.seed(1)
    for j in range(M):
        Pm[j, 0] = np.random.randint(6,12)  # 随机初始化每个服务器的处理器数
    global Cn2, Pm2
    Pm2 = Pm.copy()
    Cn2 = Cn.copy()
    for i in range(N):
        # 给任务分配副本,找到使
        for j in range(M):
            if Pm2[j,0]>=1 and Fx[0,i*M+j] >temp:
                temp = Fx[0,i*M+j]
                index = j
        Pm2[index, 0] -= 1                  # 使对应服务器资源减1
        x[i*M+index,0] = 1                  # 使对应元素值赋值为1
    print("每个任务至少有一个副本",x.T)


# 每轮迭代找到最优边际效益，使对应元素赋值为1，知道用完服务器分配资源
def DWFLA(x):
    np.random.seed(1)
    for k in range(int(Pm.sum()-N)):
        temp, index1, index2 = 0, 0, 0  # 不能放在for循环外面
        anm = x.copy()
        for i in range(N):
            for j in range(M):
                if anm[i*M+j,0] == 0:
                    anm[i*M+j,0] = 1
                else: continue
                # 判断是否满足两个约束条件
                if Cn2[i,0]>=1 and Pm2[j,0]>=1 and (f(x)-f(anm))>temp:
                    temp = f(x)-f(anm)
                    index1 = i
                    index2 = j
                anm[i * M + j, 0] = 0        # 每次计算完要使对应元素还原
        # print("第%s次迭代计算："%k)
        # print("任务%s卸载到服务器%s"%(index1+1,index2+1))
        Cn2[index1, 0] -= 1                  # 使任务剩余预算减1
        Pm2[index2, 0] -= 1                  # 使对应服务器资源减1
        x[index1*M+index2,0] = 1             # 使对应元素值赋值为1
        obj.append(Pr(x))


# 求原函数值(我们将凹函数取反求最小化）
def f(x):
    anm = x.T.copy()     # 行向量
    temp1 = 0
    for i in range(N):
        anm1 = anm[:,i*M:M+i*M]
        Fx1 = Fx[:,i*M:M+i*M]
        # print("anm1和FX1的形状:",anm1.shape,Fx1.shape)
        a = np.array(1 - Fx1) ** anm1
        temp1 += np.log(1 - np.prod(a))  # *是对应元素相乘,dot是矩阵相乘,prod将所有元素相乘
    return -temp1


def Pr(x):
    p = 1 - np.exp(-f(x))
    return p


def pr_task(x,i):    # 计算任务i完成概率Pr(Xn<=dt)
    anm = x.T.copy()  # 行向量
    anmi = anm[:,i*M:M+i*M]
    Fxi = Fx[:,i*M:M+i*M]
    temp = 1 - np.prod(np.array(1 - Fxi) ** anmi)
    return temp


def variance(x):         # 求任务的方差
    arr = np.zeros(N)
    for n in range(N):   # 计算每个任务完成概率并存入数组
        arr[n] = pr_task(x,n)
    average = np.sum(arr)/N
    var = 0
    for n in range(N):
        var += np.square(arr[n]-average)
    print("算法任务方差为:",var*1e3)


# 可视化绘图
def visual(x,y):
    # 绘制图形(采用指数分布,横坐标为外部迭代次数，纵坐标为）
    # 创建画布
    fig = plt.figure(num=1, figsize=(12, 6))
    # 字体设置为中文黑体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False  # 用来正常显示负号
    # 设置子图
    plt.subplot(111)
    plt.xlabel("外部迭代次数", fontsize=15)
    plt.ylabel("任务中断概率", fontsize=15)
    plt.title("DWFLA", fontsize=25, backgroundcolor='#3c7f99',
              fontweight='bold', color='white', verticalalignment="baseline")
    # plt.semilogy(np.arange(1, x + 1), y1, "r")
    plt.plot(np.arange(1, x + 1), y, "r")
    plt.show()


if __name__ == "__main__":
    start = datetime.datetime.now()
    init()
    DWFLA(x)
    print("最优值x:",x.T)
    print("元素之和为:",x.sum())
    print("任务中断概率为:",Pr(x))
    end = datetime.datetime.now()
    variance(x)
    print("程序运行时间: %s 秒"%(end - start).total_seconds())
    visual(int(Pm.sum()-N), obj)