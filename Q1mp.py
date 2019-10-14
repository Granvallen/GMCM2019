import xlrd
import numpy as np
import math
import sympy
from multiprocessing import Manager, Process
import time
import random

sc = time.time()

dataidx = 1
# pnum = 5
maxNse = 30

# excel文件
file = 'dataset{}.xlsx'.format(dataidx)
# 参数
if dataidx == 1:
    # 1
    a1 = 25
    a2 = 15
    b1 = 20
    b2 = 25
    theta = 30
    delta = 0.001
else:
    # 2
    a1 = 20
    a2 = 10
    b1 = 15
    b2 = 20
    theta = 20
    delta = 0.001

# 读取excel
wb = xlrd.open_workbook(filename=file)

# 所有表格的名字
sheetnames = wb.sheet_names()
sheet = wb.sheet_by_name(sheetnames[0])

# 按列获取数据
idx = sheet.col_values(0)[2:]
x = sheet.col_values(1)[2:]
y = sheet.col_values(2)[2:]
z = sheet.col_values(3)[2:]
cat = sheet.col_values(4)[3:-1]
mark = sheet.col_values(5)[3:-1]

# 分离坐标 转为向量
start = np.array([x[0], y[0], z[0]])
end = np.array([x[-1], y[-1], z[-1]])
endi = len(cat)+1 # 终点序号
x, y, z = x[1:-1], y[1:-1], z[1:-1]
vercheck = {}
horcheck = {}
hvmark = {}
for i in range(len(cat)):
    if int(cat[i]) == 1:
        vercheck[i + 1] = np.array([x[i], y[i], z[i]])
    else:
        horcheck[i + 1] = np.array([x[i], y[i], z[i]])

    hvmark[i + 1] = mark[i]


class Navigator:
    def __init__(self):
        # 记录路径
        self.lse = [] # 路线总长度
        self.Nse = [] # 经过校正点个数
        self.Lse = [] # 路径
        self.hvidxse = [] # 记录经过每个校正点时水平与竖直误差 题目要求

        self.terminatepath = set()

    def navigate(self, current, lc, Nc, hvidx, Lc, Eh, Ev, k):
        """
            current: 当前坐标
            lc: 当前为止已行进路径长度
            Nc: 当前为止已经过的校正次数
            Lc: 当前为止的路径
            Eh: 当前为止水平误差
            Ev: 当前为止垂直误差
            k: search factor

            返回在经过当前点时是否成功找到路径
        """
        if Lc == []:
            Lc.append(start)

        dist_end = np.linalg.norm(current - end)
        # 判断是否因为 该路径不可能最优 而中断搜索
        if self.prune(lc+dist_end, Nc):
            print("search terminated")
            return False

        # 这回合到达终点
        error = dist_end*delta
        if Ev + error <= theta and Eh + error <= theta:
            self.lse.append(lc + dist_end)
            self.Nse.append(Nc)
            self.Lse.append(Lc + [end])
            self.hvidxse.append(hvidx + [(endi, Eh + error, Ev + error)])
            print("route found {} Nc: {} l: {}".format(len(self.Lse), Nc, self.lse[-1]))
            return True

        # 继续搜索
        flag = False
        if not flag:
            istoend, route, route_hvidx, dist_route, idx, Ehv = self.checkroute(current, Eh, Ev)
            if istoend:
                if self.prune(lc+dist_route, Nc+1):
                    print("search terminated")
                    return False

                self.lse.append(lc + dist_route)
                self.Nse.append(Nc + 1)
                self.Lse.append(Lc + route)
                self.hvidxse.append(hvidx + route_hvidx)
                print("route found {} Nc: {} l: {}".format(len(self.Lse), Nc, self.lse[-1]))
                return True

            elif route != []:
                for i in idx:
                    print([i[0] for i in hvidx+route_hvidx[i]])
                    flag = self.navigate(route[i][1], lc+dist_route[i], Nc+2, hvidx+route_hvidx[i], Lc+list(route[i]), Ehv[i][0], Ehv[i][1], k)
            else:

                # if len(hvidx) >= 2:
                #     # print((hvidx[-2][0], hvidx[-1][0]), "banned")
                #     self.terminatepath |= {(hvidx[-2][0], hvidx[-1][0])}
                return False
        else:
            return True

    # 判断是否需要中断搜索
    def prune(self, lc, Nc):
        if Nc > maxNse:
            return True

        if self.Lse != []:
            return lc >= min(self.lse) or Nc > min(self.Nse) # 这里改成or的话是 寻找校正数与长度都要比已经找到好的路径
        else:
            return False

    def hfun(self, current, Eh, Ev, route, route_hvidx, dist_route, dist_route_end, istoend, Ehv):
        for hi, h in horcheck.items():
            dist = np.linalg.norm(current - h)
            error = dist*delta
            # 能够到达水平校正点并进行校正
            if dist != 0 and np.linalg.norm(current - end) >= np.linalg.norm(h - end) and Eh + error <= b2 and Ev + error <= b1:
                # 判断经过水平校正后能否直接到终点
                dist2 = np.linalg.norm(h - end)
                error2 = dist2*delta
                if Ev + error + error2 <= theta:
                    istoend.append((True, [h, end], [(hi, Eh+error, Ev+error), (endi, error2, Ev+error+error2)], dist+dist2, 0, [0, 0]))
                    return

                # 找接下来可达的垂直校正点
                for vi, v in vercheck.items():

                    dist3 = np.linalg.norm(h - v)
                    error3 = dist3*delta

                    if dist3 != 0 and np.linalg.norm(h - end) >= np.linalg.norm(v - end) and 0 + error3 <= a2 and Ev + error + error3 <= a1:


                        # 如果已经验证走了 这条路径后 既无法到达终点也无法到达下一个校正点 跳过路径
                        # if (hi, vi) in self.terminatepath:
                        #     continue

                        # 这一部分是当有多条路径而终点相同的情况 只保留路径最短的 加这个不会搜索加速 但能让结果更好
                        re = -1
                        for i, hv in enumerate(route_hvidx):
                            if hv[-1][0] == vi:
                                if dist_route[i] > dist + dist3:
                                    re = i
                                else:
                                    re = -2
                                break

                        if re >= 0:
                            del route[re]
                            del route_hvidx[re]
                            del dist_route[re]
                            del dist_route_end[re]
                            del Ehv[re]
                        elif re == -2:
                            continue

                        dist_end = np.linalg.norm(v - end)
                        route.append([h, v])
                        route_hvidx.append([(hi, Eh+error, Ev+error), (vi, 0+error3, Ev+error+error3)])
                        dist_route.append(dist + dist3)
                        dist_route_end.append(dist_end)
                        Ehv.append([error3, 0])
        
        istoend.append(False)
        return

    def vfun(self, current, Eh, Ev, route, route_hvidx, dist_route, dist_route_end, istoend, Ehv):

        for vi, v in vercheck.items():
            dist = np.linalg.norm(current - v)
            error = dist*delta
            # 能够到达垂直校正点并进行校正
            if dist != 0 and np.linalg.norm(current - end) >= np.linalg.norm(v - end) and Eh + error <= a2 and Ev + error <= a1:
                # 判断经过垂直校正后能否直接到终点
                dist2 = np.linalg.norm(v - end)
                error2 = dist2*delta
                if Eh + error + error2 <= theta:
                    istoend.append((True, [v, end], [(vi, Eh+error, Ev+error), (endi, error2, Ev+error+error2)], dist+dist2, 0, [0, 0]))
                    return

                # 找接下来可达的水平校正点
                for hi, h in horcheck.items():
                    dist3 = np.linalg.norm(v - h)
                    error3 = dist3*delta

                    if dist3 != 0 and np.linalg.norm(v - end) >= np.linalg.norm(h - end) and 0 + error3 <= b2 and Eh + error + error3 <= b1:

                        # 如果已经验证走了 这条路径后 既无法到达终点也无法到达下一个校正点 跳过路径
                        # if (vi, hi) in self.terminatepath:
                        #     continue

                        re = -1
                        for i, hv in enumerate(route_hvidx):
                            if hv[-1][0] == hi:
                                if dist_route[i] > dist + dist3:
                                    re = i
                                else:
                                    re = -2
                                break

                        if re >= 0:
                            del route[re]
                            del route_hvidx[re]
                            del dist_route[re]
                            del dist_route_end[re]
                            del Ehv[re]
                        elif re == -2:
                            continue

                        dist_end = np.linalg.norm(h - end)
                        route.append([v, h])
                        route_hvidx.append([(vi, Eh+error, Ev+error), (hi, Eh+error+error3, 0+error3)])
                        dist_route.append(dist + dist3)
                        dist_route_end.append(dist_end)
                        Ehv.append([0, error3])

        istoend.append(False)
        return


    def checkroute(self, current, Eh, Ev):
        """
            返回值: 
                istoend 是否到达终点 若到达终点 则其记录了到达终点路径信息 可以直接返回
                route 可行的校正路径列表(直接包含了水平与竖直校正) 
                route_hvidx 路径经过水平与竖直校正点时相应的水平竖直误差列表(题目要求)
                dist_route 路径长度列表
                idx 根据某种规则排序后元素序号列表
                Ehv 走完路径后 的(Eh, Ev)
        """

        ma = Manager()

        h_route = ma.list()
        h_dist_route = ma.list()
        h_dist_route_end = ma.list()
        h_route_hvidx = ma.list()
        h_istoend = ma.list()
        h_Ehv = ma.list()

        v_route = ma.list()
        v_dist_route = ma.list()
        v_dist_route_end = ma.list()
        v_route_hvidx = ma.list()
        v_istoend = ma.list()
        v_Ehv = ma.list()

        p1 = Process(target=self.hfun, args=(current, Eh, Ev, h_route, h_route_hvidx, h_dist_route, h_dist_route_end, h_istoend, h_Ehv))
        p2 = Process(target=self.vfun, args=(current, Eh, Ev, v_route, v_route_hvidx, v_dist_route, v_dist_route_end, v_istoend, v_Ehv))

        p1.start()
        p2.start()

        p1.join()
        p2.join()

        if h_istoend[0]:
            return h_istoend[0]

        if v_istoend[0]:
            return v_istoend[0]

        route = list(h_route) + list(v_route)
        dist_route = list(h_dist_route) + list(v_dist_route)
        dist_route_end = list(h_dist_route_end) + list(v_dist_route_end)
        route_hvidx = list(h_route_hvidx) + list(v_route_hvidx)
        Ehv = list(h_Ehv) + list(v_Ehv)

        # 到终点距离从小到大排序
        idx = sorted(range(len(dist_route_end)), key=lambda k: dist_route_end[k], reverse=False)
        # 返回6个值
        return False, route, route_hvidx, dist_route, idx, Ehv


    def show(self):
        print("len(Lse):", len(self.Lse))
        print("Nse: ", self.Nse)
        print("lse: ", self.lse)
        print("hvidxse: ", self.hvidxse)


na = Navigator()
na.navigate(current=start, lc=0, Nc=0, hvidx=[], Lc=[], Eh=0, Ev=0, k=5000)

print("--- 搜索结束 ---")
print("Time used:", (time.time() - sc))
print("a1: {}\na2: {}\nb1: {}\nb2: {}\ntheta: {}\ndelta: {}".format(a1, a2, b1, b2, theta, delta))
na.show()


# 输出
out = []
for L in na.Lse:
    out.append(list(map(list, L)))

with open('pyout{}.txt'.format(dataidx), 'w') as f:
    f.write(str(out))
