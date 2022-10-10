import numpy as np
import pandas as pd
from sklearn.cluster import KMeans as km
from sklearn.mixture import GaussianMixture as gmm
from sklearn.metrics import silhouette_score
import datetime
import os
from WindPy import w
dir = os.getcwd()
w.start()  # 默认命令超时时间为120秒，如需设置超时时间可以加入waitTime参数，例如waitTime=60,即设置命令超时时间为60秒
w.isconnected()  # 判断WindPy是否已经登录成功，如不成功请手动调整日期


def getdata(codes):  # 接受去重后的windcode，获得N日区间日均振幅，N日区间日均换手率，N日区间涨跌幅，区间相对指数涨跌幅，区间融资买入额/区间成交额，区间融资卖出额/区间成交额，需求/供给
    td = datetime.date.today().strftime("%Y%m%d")
    begin = w.tdaysoffset(-5).Data[0][0] #today-5个工作日
    begin = begin.strftime("%Y%m%d")
    data = w.wss(codes,
                 "avg_swing_per,pq_avgturn2,pct_chg_per,relpctchange,mrg_long_amt_int,amt_per,margin_shortamountint",
                 "startDate=" + begin + ";endDate=" + td + ";index=000001.SH;unit=10000;days=-5;tradeDate=" + td + ";industryType=2;ShowBlank=0",
                 usedf=True)[1]
    data.loc[:, "rq_rate"] = (data.loc[:, "MARGIN_SHORTAMOUNTINT"] / data.loc[:, 'AMT_PER']) * 100#区间融券卖出额/成交额
    data.loc[:, "rz_rate"] = (data.loc[:, "MRG_LONG_AMT_INT"] / data.loc[:, 'AMT_PER']) * 100#区间融资买入额/成交额
    #停牌标的处理
    data2 = data.drop(data.loc[data.loc[:,"AMT_PER"]==0,:].index)
    return data2


def DandS(supply, demand):
    inf = 1000000 #定义无穷大
    # 读取文件,更改此处路径即可
    today = datetime.date.today().strftime("%Y%m%d")
    yest = (datetime.date.today() + datetime.timedelta(days=-1)).strftime("%Y%m%d")
    oversupply = np.array( #获取超额供给券单
        pd.read_excel(dir+r"\机构出借券单" + today + "（客户版）.xlsx", skiprows=3, header=1,
                      dtype=({0: str})).reset_index().loc[:, "证券代码"])
    #对供给券单做透视表，获得每支标的的供给量
    s2 = supply.copy().pivot_table(index=["证券代码"], values=["股份差额"], aggfunc=[np.sum])
    s2.columns = [c2 for (c1, c2) in s2.columns.tolist()]
    s2.rename(columns={"股份差额": "supply"}, inplace=True)
    #对需求券单做透视表，获得每只标的需求量
    d2 = demand.copy().pivot_table(index=["证券代码"], values=["申请最小数量"], aggfunc=[np.sum])
    d2.columns = [c2 for (c1, c2) in d2.columns.tolist()]
    d2.rename(columns={"申请最小数量": "demand"}, inplace=True)
    #全连接上述供需表
    data = pd.concat([d2, s2], axis=1)
    data["DtoS"] = data["demand"] / data["supply"] #做商获得需求/供给
    data.loc[pd.isnull(data["supply"]), "DtoS"] = inf#supply为NA则说明无供给，则设其需求/供给为无限大
    data.loc[pd.isnull(data["demand"]), "DtoS"] = 0#supply为NA则说明无需求，则设其需求/供给为0
    data.loc[data.index.isin(oversupply), "DtoS"] = 0#在超额供给券单中的标的需求/供给全部设为0
    return data.loc[:, "DtoS"]


def z_score(data):
    data = (data - np.mean(data)) / np.std(data)
    return data


def data_process(data):#调整数据
    mx2 = data.copy().iloc[:, [0, 1, 2, 3, 7, 8, 9]] #获得指定列数据
    mx = mx2.values#转换为数组
    for i in range(np.shape(mx)[1]):#标准化
        mx[:, i] = z_score(mx[:, i])
    return mx


def data_process_drop(data):#调整数据并剔除离群点
    mx2 = data.copy().iloc[:, [0, 1, 2, 3, 7, 8, 9]]#获得指定列数据
    mx = mx2.values#转换为数组
    ots = []#离群点空表
    for i in range(np.shape(mx)[1]):
        mx[:, i] = z_score(mx[:, i])#标准化
        try:
            ots.append(np.where(np.abs(mx[:, i]) > 3)[0][0])#根据3σ原则，标准化后绝对值大于3则判断为离群点，返回对应样本位置
        except:
            continue
    ots = list(set(ots))#去重
    mx = np.delete(mx, ots, axis=0)#删除离群点对应样本
    return [mx, ots]


# 异常值label合并为排名大于其且距离最近的label
def combine(dist, ot, label):
    poss = [np.where(dist == i)[0][0] for i in ot] #离群点簇所在dist_sort位置
    to = []#离群点簇转变为的label
    flag = -1 #控制label匹配方向，默认为左
    for i in poss:
        p0 = i - 1  #离群点簇优先合并为其左侧label，及欧氏距离大于其的label，对应更高的费率
        while ((p0) in poss) | (p0 == -1):#当左侧簇为离群点簇时继续向左移动，若位置为-1则向右移动
            if (p0 == -1):
                flag = 1
            p0 = p0 + flag * 1
        to.append(dist[p0]) #最终合并变化的结果
    for i in range(ot.shape[0]):
        label.loc[label.loc[:, "label"] == ot[i], "label"] = to[i]#修改离群点label值为对应的合并后的label
    dist = np.delete(dist, poss)#删除原簇距离排名中的离群点簇
    return [label, dist]

def rate_pair(score, labels, centers):#假设离群点将被单独聚类，故在最优的聚类结果寻找由离群点构成的簇，将其中的离群点簇合并到其他簇中，最后按其欧氏距离匹配费率

    max_n = np.argmax(score) #获得评分最高的训练结果的位置
    max_label, max_center = labels[max_n], centers[max_n] #获得最高评分的训练结果
    ot = outlier(max_label) #最优结果中的
    ot_point = ot[ot == True].axes[0].asi8 #返回离群点簇位置
    mc = np.mat(max_center) #矩阵化
    max_center.loc[:, "E_dist"] = [np.sqrt((mc[i] * mc[i].T)[0, 0]) for i in range(mc.shape[0])]  # 计算各簇心距原点欧式距离
    dist_sort = np.argsort(-max_center.loc[:, "E_dist"].values) #返回按欧式距离降序排列的max_center中各簇的位置
    max_center.loc[:, "rate"] = [0.0] * dist_sort.shape[0] #新建rate column

    # 离群点簇合并
    [max_label2, dist2] = combine(dist_sort, ot_point, max_label)
    for i in range(dist2.shape[0]):
        max_center.loc[:, "rate"][dist2[i]] = 7.5 - 0.5 * i #按降序排名赋予每个簇对应的费率，i从0-5，如dist2[0]为对应最高费率的簇的位置
    label_rate = pd.merge(max_label2, max_center.loc[:, "rate"], left_on="label", right_index=True)#合并label表和对应的费率
    label_rate.loc[:, "code"] = [i[:6] for i in label_rate.loc[:, 0]] #windcode转变为6位代码
    label_rate.set_index("code", inplace=True)
    label_rate = label_rate.loc[:, "rate"]
    return label_rate


def rate_pair_drop(score, labels, centers, ots):
    max_n = np.argmax(score)#获得评分最高的训练结果的位置
    max_label, max_center = labels[max_n], centers[max_n]#获得最高评分的训练结果
    mc = np.mat(max_center)#最优结果中的
    max_center.loc[:, "E_dist"] = [np.sqrt((mc[i] * mc[i].T)[0, 0]) for i in range(mc.shape[0])]  # 各中心距原点欧式距离
    dist_sort = np.argsort(-max_center.loc[:, "E_dist"].values)  # 返回按欧式距离降序排列的max_center中各簇的位置
    max_center.loc[:, "rate"] = [0.0] * dist_sort.shape[0]  # 新建rate column
    for i in range(dist_sort.shape[0]):
        max_center.loc[:, "rate"][dist_sort[i]] = 7.5 - 0.5 * i  #按降序排名赋予每个簇对应的费率，i从0-5，如dist2[0]为对应最高费率的簇的位置
    for i, row in ots.iterrows():
        max_label = max_label.append([{0: i, "label": max_center.index[dist_sort[0]]}], ignore_index=False) #排除于训练集外的样本重新添加回来，由离群点的筛选规则直接赋予其最高费率对应的label
    label_rate = pd.merge(max_label, max_center.loc[:, "rate"], left_on="label", right_index=True)#合并label表和对应的费率
    label_rate.loc[:, "code"] = [i[:6] for i in label_rate.loc[:, 0]]#windcode转变为6位代码
    label_rate.set_index("code", inplace=True)
    label_rate = label_rate.loc[:, "rate"]
    return label_rate


def outlier(label, a=0.01):
    label.loc[:, "anything"] = 1 #用于统计每个簇的样本数
    la = label.groupby(label["label"]).count()
    la2 = la["anything"] < label.shape[0] * a #每个簇中样本数小于样本总量*0.01时判断其中样本为离群点
    return la2

#将离群点单独分为一类用于处理离群点
def kmeans(data,clusters = 5):
    mx = data_process(data) #处理数据
    model = km(n_clusters=clusters, max_iter=100000) #第一次训练用于判断离群点
    model.fit(mx)
    label = pd.DataFrame(model.labels_, columns=["label"])
    clusters += outlier(label).sum() #若发现离群点，则在cluster上加入离群点数量以在之后的训练中将离群点单独聚类
    model = km(n_clusters=clusters, max_iter=100000)
    model.fit(mx)

    #存储训练结果
    label = pd.DataFrame(model.labels_,columns=["label"])
    codes = pd.DataFrame(data.index)
    label= pd.concat([codes,label],axis=1)
    labels = [label]
    center = pd.DataFrame(model.cluster_centers_)
    centers = [center]
    score= [silhouette_score(mx, model.labels_, metric='euclidean')] #轮廓系数

    for i in range(50):    #总计51次聚类结果统计
        model = km(n_clusters=clusters,max_iter=100000)
        model.fit(mx)
        label = pd.DataFrame(model.labels_, columns=["label"])
        score.append(silhouette_score(mx, model.labels_, metric='euclidean'))
        label = pd.concat([codes, label], axis=1)
        labels.append(label)
        center = pd.DataFrame(model.cluster_centers_)
        centers.append(center)
    return rate_pair(score,labels,centers) #聚类结果选择及费率判断

#根据3σ原则剔除离群点使其不参与训练，在最后直接赋予离群点最高的费率
def kmeans_drop(data, clusters=5):
    [mx, ots] = data_process_drop(data) #处理数据及返回离群点在样本集中的位置（第多少行）以及提出离群点后的训练集
    ots_data = data.iloc[ots, :] #提取离群点数据
    data = data.drop(ots_data.index) #删除用于训练的数据中的离群点
    model = km(n_clusters=clusters, max_iter=100000)
    model.fit(mx)

    #存储训练结果
    label = pd.DataFrame(model.labels_, columns=["label"])
    codes = pd.Series(data.index)
    label = pd.concat([codes, label], axis=1)
    labels = [label]
    center = pd.DataFrame(model.cluster_centers_)
    centers = [center]
    score = [silhouette_score(mx, model.labels_, metric='euclidean')]
    for i in range(50):  # 总计51次聚类结果统计
        model = km(n_clusters=clusters, max_iter=100000)
        model.fit(mx)
        label = pd.DataFrame(model.labels_, columns=["label"])
        score.append(silhouette_score(mx, model.labels_, metric='euclidean'))
        label = pd.concat([codes, label], axis=1)
        labels.append(label)
        center = pd.DataFrame(model.cluster_centers_)
        centers.append(center)
    return rate_pair_drop(score, labels, centers, ots_data)#聚类结果选择及费率判断

#GMM聚类，不剔除离群点，具体算法与kmeans一致
def GMM(data, clusters=5):
    mx = data_process(data)
    ots = np.empty((0, 10))
    for i in range(10):
        model = gmm(n_components=clusters, max_iter=100000)
        model.fit(mx)
        label = model.predict(mx).reshape([-1, 1])
        label2 = pd.DataFrame(label, columns=["label"])
        ots = np.append(ots, outlier(label2).sum())
    clusters += int(ots.max())
    model = gmm(n_components=clusters, max_iter=100000)
    model.fit(mx)
    label = model.predict(mx).reshape([-1, 1])
    score = [-1*model.aic(mx)]
    codes = pd.DataFrame(data.index)
    label2 = pd.DataFrame(label, columns=["label"])
    label = pd.concat([codes, label2], axis=1)
    labels = [label]
    center = pd.DataFrame(model.means_)
    centers = [center]
    for i in range(50):  # 总计51次聚类结果统计
        model = gmm(n_components=clusters, max_iter=100000)
        model.fit(mx)
        label = model.predict(mx).reshape([-1, 1])
        score.append(-1*model.aic(mx))
        label2 = pd.DataFrame(label, columns=["label"])
        label = pd.concat([codes, label2], axis=1)
        labels.append(label)
        center = pd.DataFrame(model.means_)
        centers.append(center)

    return rate_pair(score, labels, centers)

#GMM聚类，剔除离群点，具体算法与kmeans一致
def GMM_drop(data, clusters=5):
    [mx, ots] = data_process_drop(data)  # 处理数据及返回离群点在样本集中的位置（第多少行）以及提出离群点后的训练集
    ots_data = data.iloc[ots, :]  # 提取离群点数据
    data = data.drop(ots_data.index)  # 删除用于训练的数据中的离群点

    model = gmm(n_components=clusters, max_iter=100000)
    model.fit(mx)

    #存储训练结果
    label = model.predict(mx).reshape([-1, 1])
    score = [-1*model.aic(mx)]
    codes = pd.DataFrame(data.index)
    label2 = pd.DataFrame(label, columns=["label"])
    label = pd.concat([codes, label2], axis=1)
    labels = [label]
    center = pd.DataFrame(model.means_)
    centers = [center]
    for i in range(50):  # 总计51次聚类结果统计
        model = gmm(n_components=clusters, max_iter=100000)
        model.fit(mx)
        label = model.predict(mx).reshape([-1, 1])
        score.append(-1*model.aic(mx))
        label2 = pd.DataFrame(label, columns=["label"])
        label = pd.concat([codes, label2], axis=1)
        labels.append(label)
        center = pd.DataFrame(model.means_)
        centers.append(center)

    return rate_pair_drop(score, labels, centers,ots_data)