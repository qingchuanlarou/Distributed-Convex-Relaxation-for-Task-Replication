# 指数分布，在GA3的基础上将初始化部分由服务器轮循改为用户任务轮循，避免任务中断概率为1，
# 加入了连续三次迭代最佳个体不变就停止迭代
import math, random
import numpy as np  # 二维数组对某一行切片是生成一维向量而不是一维数组
import datetime
import scipy.special
import winsound
from matplotlib import pyplot as plt

# np.set_printoptions(threshold=1000000)  # 元素门槛值。数组个数沒有超过设置的阈值，NumPy就会将Python將所有元素列印出來。

N = 40  # 任务数
M = 10  # 服务器数
chrom_size = N * M  # 染色体长度
popsize = 100  # 种群的个体数量
maxgen = 50  # 进化最大世代数
Nselec = int(popsize * 0.05)  # 浮点数，作为索引要转换为整数
Nco = int(popsize * 0.04)  #
C1 = np.zeros((N, N * M))  # 不等式1的系数矩阵，N行（N*M）列
C2 = np.zeros((M, N * M))  # 不等式2的系数矩阵，M行（N*M)列
Cn = np.zeros((1, N))  # 每个子任务的总成本限制，
Cnm = np.ones((N, M))  # 单个副本的成本，均为1
obj = []
tau = 1              # 期望阈值为1,去掉了公共延迟分量,这是总的时间，传输时间其实更小
count = np.zeros(1)  # 计算器，连续五次最佳个体不更新则停止迭代
# 随机初始化Fx=Pr(Ynm<=tau)值

np.random.seed(1)  # 使用相同的seed()值，保证同一程序每次运行结果一样(设置的seed()值仅一次有效)
B = 1e6  # 用户与服务器之间带宽均设置为1Mhz，10的6次方
gamma = np.random.uniform(1, 30, size=N * M).reshape((N, M))  # 信噪比取值为[1,30),[1,N*M]
lambert = np.real(scipy.special.lambertw(gamma, k=0, tol=1e-8))  # 信噪比的朗伯w函数,np.real()取实部
r = B * lambert / np.log(2)  # 传输速率约为几Mb/s
np.random.seed(1)
bn = np.random.randint(4e5, 1e6, size=N)  # 任务传输大小0.4Mb-1Mb
bnm = np.zeros((N, M))  # 方便计算将任务传输大小扩展为N*M维
for i in range(N):
    for j in range(M):
        bnm[i, j] = bn[i]
w = r / bnm * np.exp(-1 / gamma * (2 ** (r / B) - 1))  # 服务速率w表达公式
np.random.seed(1)
dn = np.random.uniform(0.1, 0.8, size=N)  # 任务所需要计算量，0.1G-0.8G个CPU周期
dnm = np.zeros((N, M))
for i in range(N):
    for j in range(M):
        dnm[i, j] = dn[i]
np.random.seed(1)
Fm = np.random.randint(14, 20, size=M)  # 随机初始化服务器计算能力,多核总和(GHz)
fm = 1  # 服务器m给单个给单个任务分配的频率，这里假设所有服务器相同,1GHZ
Pm = (Fm / fm).reshape((1, M))  # 服务器m最大能并行执行的任务副本数
Fx = np.array(1 - np.exp(-1 * w * (tau - dnm / fm))).reshape((1, N * M))  # Ynm的累积分布函数，Pr(Ynm<=dt)的取值
arg_Fx = np.argsort(Fx)  # 增序排序提取索引
# tau-dnm/fm必须要保证其值大于0，否则FX变为复数
print("信噪比为为:", gamma)
print("朗伯w函数为：", lambert)
print("传输速率速率为：", r)
print("任务传输数据大小(bits):", bnm)
print("服务速率:", w)
print("服务速率w增序取值：", np.sort(w, axis=None))
print("任务执行所需要周期:", dnm)  # 指数分布
print("dnm/fm增序取值:", np.sort(dnm / fm, axis=None))
print("任务完成概率Pr(Ynm<=dt):", Fx)
print("任务完成概率Pr(Ynm<=dt)增序取值:", np.sort(Fx, axis=None))
print("服务器数量:", Pm, np.sum(Pm))


class Population:
    # 种群的设计
    def __init__(self, p, c, Ns, Nc, m):
        # 初始化约束条件
        for i in range(N):
            for j in range(M):
                C1[i][j + i * M] = Cnm[i][j]  # 使非N所在系数为0
        for i in range(M):
            for j in range(N):
                C2[i][i + j * M] = fm  # 使非M所在系数为0
        np.random.seed(1)
        for i in range(N):
            Cn[0, i] = np.random.randint(5, 20)  # 随机初始化每个子任务n的成本
        print("任务资源约束:", Cn, np.sum(Cn))

        # 种群信息合集
        self.individuals = np.zeros((p, c))  # 个体集合,每一行为一个个体
        self.fitness = np.ones(p)  # 个体适应度集
        self.new_individuals = np.zeros((p, c))  # 新一代个体集合
        self.new_fitness = np.ones(p)  # 新一代个体适应度集

        self.popsize = p  # 种群所包含的个体数
        self.chromosome_size = c  # 个体的染色体长度
        self.Nselec = Ns  # 亲本对数
        self.Nco = Nc  # 每一对亲本产生的子代个数

        self.generation_max = m  # 种群进化的最大世代数
        self.gen = 0             # 种群停止时所处世代

        self.elitist = np.zeros(2)  # 最佳个体的信息[种群中位置，适应度，第几代]
        self.elitist[1] = 1         # 使自适应度(任工作中断概率)初始化为1
        print("初始化参数")

    # 条件随机搜索(CRS)算法,用在初始化
    def ini_population(self):
        arr = np.arange(M)
        np.random.seed(0)
        for i in range(self.popsize):  # 每次循环随机产生一个可行个体
            k = 0
            while True:
                Flag = True
                for n in range(N):     # N个用户轮循赋副本
                    np.random.shuffle(arr)
                    for j in range(M):    # 最多M次可找到Xnm=0的随机索引
                        if self.individuals[i, n*M+int(arr[j])] == 1:
                            # print("循环1")
                            continue
                        else:
                            self.individuals[i, n*M+int(arr[j])] = 1
                            # 当原始整数解满足条件时
                            if np.all((Cn - np.dot(C1, self.individuals[i, :])) >= 0) and np.all(
                                    (Pm - np.dot(C2, self.individuals[i, :])) >= 0):
                                #print("当前元素赋1")
                                k += 1
                                Flag = False
                                break
                            else:
                                self.individuals[i, n*M+int(arr[j])] = 0
                                continue
                # print("K=", k)
                # print("任务副本个数:",np.dot(C1, self.individuals[i, :]))
                # print("约束条件1:", Cn - np.dot(C1, self.individuals[i, :]))
                # print("约束条件2:", Pm - np.dot(C2, self.individuals[i, :]))
                # 某一次while循环只要副本没增加(Flag=True)也退出循环
                if k == min(np.sum(Cn), np.sum(Pm)) or Flag:
                    break
            self.fitness[i] = self.fitness_func(self.individuals[i, :])
            print("产生第%d个个体,副本数为%d"%(i,k),self.individuals[i, :])
        print("种群初始化完成!")

    # 求原凸函数值(我们将凹函数取反求最小化）
    def f1(self, x):
        anm = x.copy()  # 别忘了后面加一对圆括号
        temp1 = 0
        # print("anm:",anm)
        for i in range(N):
            anm1 = anm[i * M:M + i * M]
            Fx1 = Fx[0, i * M:M + i * M]
            a = np.array(1 - Fx1) ** anm1  # Fx1为底，anm1为幂
            temp1 += np.log(1 - np.prod(a))  # *是对应元素相乘,dot是矩阵相乘,prod将所有元素相乘
        return -temp1

    # 单个个体的适应度(工作中断概率，越小越好)
    def fitness_func(self, x):
        '''适应度函数，每一个个体(可行解)有自己的适应度(工作中断概率)'''
        p = 1 - np.exp(-self.f1(x))
        return p

    def pr_task(self,x, i):  # 计算任务i完成概率Pr(Xn<=dt)
        anm = x.copy().reshape((1,N*M))       # 去掉转置，传入的就是行向量
        anmi = anm[:, i * M:M + i * M]
        Fxi = Fx[:, i * M:M + i * M]
        temp = 1 - np.prod(np.array(1 - Fxi) ** anmi)
        return temp

    def variance(self, x):  # 求任务的方差
        arr = np.zeros(N)
        for n in range(N):  # 计算每个任务完成概率并存入数组
            arr[n] = self.pr_task(x, n)
        average = np.sum(arr) / N
        print("平均值为:",average)
        var = 0
        for n in range(N):
            var += np.square(arr[n] - average)
        print("算法任务方差为:", var * 1e3)


    # 用于评估种群中的个体集合 self.individuals 中各个个体的适应度
    def evaluate(self):
        '''用于评估种群中的个体集合 self.individuals 中各个个体的适应度'''
        for i in range(self.popsize):  # 将计算结果保存在 self.fitness列表中
            # print(self.individuals[i,:])
            self.fitness[i] = self.fitness_func(self.individuals[i, :])  # 传入参数为一维向量

    # 条件随机搜索(CRS)算法，用在变异
    def CRS(self, st, it):
        arr = np.arange(N)
        np.random.seed(it)
        for i in range(self.popsize - st):  # 每次循环随机产生一个可行个体
            k = 0
            while True:
                Flag = True
                for m in range(M):          # M个服务器轮循赋副本
                    np.random.shuffle(arr)
                    for j in range(N):      # 最多N次可找到Xnm=0的随机索引
                        if self.new_individuals[i + st, int(arr[j])*M+m] == 1:
                            # print("循环1")
                            continue
                        else:
                            self.new_individuals[i + st, int(arr[j])*M+m] = 1
                            # 当原始整数解满足条件时
                            if np.all((Cn - np.dot(C1, self.new_individuals[i + st, :])) >= 0) and np.all(
                                    (Pm - np.dot(C2, self.new_individuals[i + st, :])) >= 0):
                                # print("当前元素赋1")
                                k += 1
                                Flag = False
                                break
                            else:
                                # print("循环2")
                                # print("约束1:",Cn - np.dot(C1, self.individuals[i+st, :]))
                                # print("约束2:",Pm - np.dot(C2, self.individuals[i+st, :]))
                                self.new_individuals[i + st, int(arr[j])*M+m] = 0
                                continue
                # print("K=",k)
                # 某一次while循环只要副本没增加(Flag=True)也退出循环
                if k == min(np.sum(Cn), np.sum(Pm)) or Flag:
                    break
            self.new_fitness[i + st] = self.fitness_func(self.new_individuals[i + st, :])
        print("已完成第%d代变异操作" % it)

    # 保留最佳个体
    def reproduct_elitist(self):
        # 与当前种群进行适应度比较，更新最佳个体
        i = np.argmin(self.fitness)     # 找工作中断概率最小的个体
        if self.fitness[i] < self.elitist[1]:
            self.elitist[0] = i
            self.elitist[1] = self.fitness[i]
        for j in range(chrom_size):     # 将最佳个体加入下一代种群并更新自适应度
            self.new_individuals[0, j] = self.individuals[int(self.elitist[0]), j]
        self.new_fitness[0] = self.elitist[1]

    # 交叉
    def crossover(self, it):
        update_ind = 0  # 记录更新种群的个体索引
        # arg = np.argsort(Fx)      # 产生新数组，任务完成概率小的排在前面
        # arg_n = np.zeros((N, M))  # 每一行用户的任务副本按照增序排序
        # arg_m = np.zeros((M, N))  # 每一行服务器连接的任务按照增序排序
        # n_t = np.zeros(N)         # 值为当前索引用户对应的副本已排好序的数量
        # m_t = np.zeros(M)         # 值为当前索引服务器对应的副本已排好序的数量
        # for k in range(N * M):
        #     n = arg[0, k] // M    # 确定k在数组哪一行,即哪个任务(求商)
        #     m = arg[0, k] % M     # 确定k在数组哪一列,即哪个服务器(求余)
        #     arg_n[n, int(n_t[n])] = arg[0, k]
        #     arg_m[m, int(m_t[m])] = arg[0, k]
        #     n_t[n] += 1
        #     m_t[m] += 1

        self.reproduct_elitist()  # 将最佳个体加入下一代种群

        np.random.seed(it)
        arg_fit = np.argsort(self.fitness)
        arg1 = arg_fit[1:self.Nselec * 2 + 1].copy()  # 找出中断概率最低的前2*Nselec个索引(除去最佳个体)
        np.random.shuffle(arg1)  # 打乱数组，前一部分作为副本，后一部分作为母本
        a = arg1[:self.Nselec]  # 父本
        b = arg1[self.Nselec:]  # 母本
        # print("arg1=",arg1)
        # print("a=",a)
        # print("b=",b)
        for i in range(self.Nselec * 2):  # 将这2*Nselec个个体加入下一代种群
            for j in range(chrom_size):
                self.new_individuals[i + 1, j] = self.individuals[arg1[i], j]
            self.new_fitness[i + 1] = self.fitness[arg1[i]]
        update_ind += self.Nselec * 2 + 1  # 更新种群更新的索引数

        for i in range(self.Nselec):
            # 对当前亲本，找到Nco个不重复的分裂点
            breakpoint = np.random.choice(self.popsize - 1, self.Nco, replace=False) + 1
            # print("分裂点集合:",breakpoint)
            for j in range(self.Nco):
                # a中个体取breakpoint前面的基因+b中个体breakpoint即以后的基因拼接成一个新的个体,反之组成另一个新的个体
                chrome1 = np.append(self.individuals[a[i], :breakpoint[j]], self.individuals[b[i], breakpoint[j]:])
                chrome2 = np.append(self.individuals[b[i], :breakpoint[j]], self.individuals[a[i], breakpoint[j]:])
                # print("个体形状为:",chrome1.shape)
                # 判断这两个个体是否满足约束条件,如果不满足则使其满足条件
                arr1 = np.arange(N)
                arr2 = np.arange(M)
                if not np.all((Pm - np.dot(C2, chrome1)) >= 0):
                    for m in range(M):  # 找出超出资源的服务器并使其满足约束条件
                        np.random.shuffle(arr1)
                        if (Pm[0, m] - np.dot(C2, chrome1)[m]) < 0:
                            # 超出k个资源就要丢掉k个任务，每次在选中的副本中随机选择赋值为0
                            k = np.dot(C2, chrome1)[m] - Pm[0, m]
                            for n in range(N):
                                if chrome1[int(arr1[n]*M+m)] == 1:
                                    chrome1[int(arr1[n]*M+m)] = 0
                                    k -= 1
                                if k == 0:
                                    break
                if not np.all((Cn - np.dot(C1, chrome1)) >= 0):
                    for n in range(N):  # 找出超出预算的任务并使其满足约束条件
                        np.random.shuffle(arr2)
                        if (Cn[0, n] - np.dot(C1, chrome1)[n]) < 0:
                            # 超出多少预算就执行多少次，每次在选中的副本中随机选择赋值为0
                            k = np.dot(C1, chrome1)[n] - Cn[0, n]
                            for m in range(M):
                                if chrome1[int(n*M+arr2[m])] == 1:
                                    chrome1[int(n*M+arr2[m])] = 0
                                    k -= 1
                                if k == 0:
                                    break
                if not np.all((Pm - np.dot(C2, chrome2)) >= 0):
                    for m in range(M):  # 找出超出资源的服务器并使其满足约束条件
                        np.random.shuffle(arr1)
                        if (Pm[0, m] - np.dot(C2, chrome2)[m]) < 0:
                            # 超出k个资源就要丢掉k个任务，每次在选中的副本中随机选择赋值为0
                            k = np.dot(C2, chrome2)[m] - Pm[0, m]
                            for n in range(N):
                                if chrome2[int(arr1[n]*M+m)] == 1:
                                    chrome2[int(arr1[n]*M+m)] = 0
                                    k -= 1
                                if k == 0:
                                    break
                if not np.all((Cn - np.dot(C1, chrome2)) >= 0):
                    for n in range(N):  # 找出超出预算的任务并使其满足约束条件
                        np.random.shuffle(arr2)
                        if (Cn[0, n] - np.dot(C1, chrome2)[n]) < 0:
                            # 超出多少预算就执行多少次，每次在选中的副本中随机选择赋值为0
                            k = np.dot(C1, chrome2)[n] - Cn[0, n]
                            for m in range(M):
                                if chrome2[int(n*M+arr2[m])] == 1:
                                    chrome2[int(n*M+arr2[m])] = 0
                                    k -= 1
                                if k == 0:
                                    break

                # 判断是否有服务器没利用完,有则每次随机增加任务副本，同时需要判断是否满足预算约束
                if np.any(Pm - np.dot(C2, chrome1)) > 0:  # 存在True则返回True
                    for m in range(M):  # 找出超出资源的服务器并使其满足约束条件
                        np.random.shuffle(arr1)
                        if (Pm[0, m] - np.dot(C2, chrome1)[m]) > 0:
                            # 剩余k个资源就要产生新的k个任务，每次在选中的副本中选择任务完成概率最大的赋值为1
                            k = Pm[0, m] - np.dot(C2, chrome1)[m]
                            for n in range(N):
                                if chrome1[int(arr1[n] * M + m)] == 0:
                                    chrome1[int(arr1[n] * M + m)] = 1
                                    if (Cn[0, int(arr1[n])] - np.dot(C1, chrome1)[int(arr1[n])]) < 0:  # 判断增加后是否满足约束1
                                        chrome1[int(arr1[n] * M + m)] = 0
                                        continue
                                    k -= 1
                                if k == 0:
                                    break
                if np.any(Pm - np.dot(C2, chrome2)) > 0:  # 存在True则返回True
                    for m in range(M):  # 找出超出资源的服务器并使其满足约束条件
                        np.random.shuffle(arr1)
                        if (Pm[0, m] - np.dot(C2, chrome2)[m]) > 0:
                            # 剩余k个资源就要产生新的k个任务，每次在选中的副本中选择任务完成概率最大的赋值为1
                            k = Pm[0, m] - np.dot(C2, chrome2)[m]
                            for n in range(N):
                                if chrome2[int(arr1[n] * M + m)] == 0:
                                    chrome2[int(arr1[n] * M + m)] = 1
                                    if (Cn[0, int(arr1[n])] - np.dot(C1, chrome2)[int(arr1[n])]) < 0:  # 判断增加后是否满足约束1
                                        chrome2[int(arr1[n] * M + m)] = 0
                                        continue
                                    k -= 1
                                if k == 0:
                                    break

                # 将两个子本加入下一代种群
                for t in range(chrom_size):
                    self.new_individuals[update_ind, t] = chrome1[t]
                    self.new_fitness[update_ind] = self.fitness_func(chrome1)
                    self.new_individuals[update_ind + 1, t] = chrome2[t]
                    self.new_fitness[update_ind + 1] = self.fitness_func(chrome2)
                update_ind += 2
        # print("update_ind=",update_ind)
        print("已完成第%d代交叉操作" % it)

    # 变异
    def mutation(self, it):
        self.CRS(1 + 2 * self.Nselec + 2 * self.Nselec * self.Nco, it)

    # 进化过程
    def evolve(self, it):
        # 计算适应度
        self.evaluate()
        # 选择两个个体，进行交叉与变异，产生新的种群
        self.crossover(it)
        self.mutation(it)
        # 将种群和自适应度进行更新
        for i in range(self.popsize):
            for j in range(self.chromosome_size):
                self.individuals[i, j] = self.new_individuals[i, j]
                self.new_individuals[i, j] = 0
            self.fitness[i] = self.new_fitness[i]
            self.new_fitness[i] = 0
        # print("第%d代种群的适应度集:" % it, self.fitness)
        ind = np.argmin(self.fitness)
        if ind == 0:
            print("最佳个体为上代继承")
            count[0] +=1
        elif ind < (1 + 2 * self.Nselec + 2 * self.Nselec * self.Nco):
            print("最佳个体为交叉操作所得")
            count[0] = 0    # 要连续3次没有优化
        else:
            print("最佳个体为变异所得")
            count[0] = 0
        obj.append(self.fitness[ind])      # 每次迭代记录最佳个体工作中断概率

    def run(self):
        # 初始化生成第一代种群
        self.ini_population()
        print("初始化种群中个体最好适应度:%f, 平均适应度:%f, 最差适应度:%f" % (
            np.min(self.fitness), np.sum(self.fitness) / self.popsize, np.max(self.fitness)))
        # print("初始化种群为:",self.individuals)
        '''根据种群最大进化世代数设定了一个循环。
        在循环过程中，调用 evolve 函数进行种群进化计算，并输出种群的每一代的个体适应度最大值、平均值和最小值。'''
        for i in range(1, self.generation_max + 1):
            self.evolve(i)
            print("第%d代:" % i)
            print("个体最好适应度:%f, 平均适应度:%f, 最差适应度:%f" % (
            np.min(self.fitness), np.sum(self.fitness) / self.popsize, np.max(self.fitness)))
            if count[0] == 5:
                self.gen = i
                break
        print("进化完成!")
        ind = np.argmin(self.fitness)
        print("资源和为",np.sum(Pm))
        print("最佳个体为:", self.individuals[ind, :])
        print("元素求和为:",np.sum(self.individuals[ind, :]))
        print("约束条件1,Cn-A1@x.T=", (Cn - np.dot(C1, self.individuals[ind, :])))
        print("约束条件2,Pm-A2@x.T=", (Pm - np.dot(C2, self.individuals[ind, :])))
        print("工作中断概率为:", self.fitness[ind])
        self.variance(self.individuals[ind, :])

    # 可视化绘图
    def visual(self):
        # 绘制图形(采用指数分布,横坐标为外部迭代次数，纵坐标为）
        # 创建画布
        fig = plt.figure(num=1, figsize=(12, 6))
        # 字体设置为中文黑体
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        # 设置子图
        plt.subplot(111)
        plt.title("GA_Algorithm", fontsize=25, backgroundcolor='#3c7f99',
                  fontweight='bold', color='white', verticalalignment="baseline")
        plt.xlabel("Number of evolution", fontsize=15)
        plt.ylabel("Job outage probability", fontsize=15)
        # plt.semilogy(np.arange(1, x + 1), y1, "r")
        plt.plot(np.arange(1, self.gen + 1), obj, "r")
        #plt.savefig('data/1.4/GA.png')
        plt.show()


if __name__ == '__main__':
    start = datetime.datetime.now()
    # 种群的个体数量为 50，染色体长度为N*M(变量维度),进化最大世代数为 50
    pop = Population(popsize, chrom_size, Nselec, Nco, maxgen)
    pop.run()
    end = datetime.datetime.now()
    print('程序运行时间: %s 秒' % (end - start).total_seconds())
    duration = 3000         # 持续时间以毫秒为单位，这里是5秒
    freq = 440              # Hz
    winsound.Beep(freq, duration)
    #np.savetxt('data/1.4/GA-obj.txt', obj, fmt="%.4f")
    pop.visual()