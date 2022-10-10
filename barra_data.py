import sys
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from WindPy import w
import statsmodels as sm
import os

w.start() # 默认命令超时时间为120秒，如需设置超时时间可以加入waitTime参数，例如waitTime=60,即设置命令超时时间为60秒
w.isconnected() # 判断WindPy是否已经登录成功，如不成功请手动调整日期

prices = pd.read_excel(r"E:\research\exposure\CNE5_Daily_Asset_Price.xlsx",dtype={"code":str})
prices = prices[~prices["code"].duplicated()]#去重
asset_data = pd.read_excel(r"E:\research\exposure\CNE5S_100_Asset_Data.xlsx",dtype={"code":str})
asset_data = asset_data[~asset_data["code"].duplicated()] #去重
exps = pd.read_excel(r"E:\research\exposure\CNE5S_100_Asset_Exposure.xlsx",dtype={"code":str})
covs = pd.read_excel(r"E:\research\exposure\CNE5S_100_Covariance.xlsx")

#返回指定因子协方差矩阵
def cov_mx(factor_ids): #factor_ids 因子名字符串列表
    cov = []
    for i in factor_ids:
        fs = []
        for j in factor_ids:
            try:
                f_cov = covs.loc[(covs['!Factor1'] == i)&(covs['Factor2'] == j),"VarCovar"].values[0]
            except:
                f_cov = covs.loc[(covs['!Factor1'] == j) & (covs['Factor2'] == i), "VarCovar"].values[0]
            fs.append(f_cov)
        cov.append(fs)
    return cov

#返回输入标的specific risk
def Get_spec_risk(asset_ids):
    spec_risk = []
    for ids in asset_ids:
        spec_risk.append(asset_data.loc[asset_data["code"]==ids,'SpecRisk%'].values[0])
    return spec_risk

#返回输入标的price
def Get_price(asset_ids):
    asset_price = []
    for ids in asset_ids:
        asset_price.append(prices.loc[prices["code"] == ids, 'Price'].values[0])
    return asset_price
#返回标的因子exposure
def Get_Exposures(asset_ids,factor_ids):
    asset_exp = []
    for i in asset_ids:
        spec_exp = []
        exp_p = exps[exps["code"]==i]
        for j in factor_ids:
            try:
                spec_exp.append(exp_p.loc[exps['Factor']==j,'Exposure'].values[0])
            except:
                spec_exp.append(0) #industry factor
            #print(np.mat(asset_exp))
        asset_exp.append(spec_exp)
    return asset_exp


factor = covs['!Factor1']
factor = factor[~factor.duplicated()][:-1].tolist()
factor = factor




def track_error(mng_id,bmk_id,mng_weight,bmk_weight=[],pic_name = '0',basevalue = 1000000,dirname = "default"): #only for active instance
    mng_weight = mng_weight["value"].tolist()[2:]
    if np.sum(mng_weight[:len(mng_id)])<0:
        bmk_weight = [0]*len(mng_id)+mng_weight[-len(bmk_id):]
        mng_weight = mng_weight[:len(mng_id)]+[0]*len(bmk_id)
    begin = "20211101"
    end = "20211123"
    id = mng_id+bmk_id
    code_str = ",".join(id)
    price = Get_price(id)
    mng_shares = np.mat([round(basevalue*mng_weight[i]/price[i]) for i in range(len(id))])
    bmk_shares = np.mat([round(basevalue*bmk_weight[i]/price[i]) for i in range(len(id))])
    price_series = w.wsd(code_str, "close", begin, end, "",usedf=True)[1]
    price = np.mat(price_series.values)
    #value
    mng_v = price*mng_shares.T
    bmk_v = price*bmk_shares.T
    hedged_s = bmk_shares.T[-1]
    hedged_v = price[:,-1]*hedged_s
    replication = mng_v-bmk_v+hedged_v
    mng_r = mng_v / basevalue
    bmk_r = bmk_v / basevalue
    #ploting
    x = [i for i in range(len(mng_v))]
    plt.plot(x, mng_r-bmk_r, ":", label="tracking_error", color="black")
    plt.legend()
    seconds = dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    dir = r"E:\pycharm\barra\\"+dirname
    if not os.path.exists(dir):
        os.makedirs(dir)
    plt.savefig(dir+r"\\"+pic_name+seconds+".jpg")
    plt.show()
    plt.close()
    #ploting replication
    plt.plot(x, replication, ":", label="replica", color="black")
    plt.plot(x, hedged_v, ":", label="replica", color="red")
    plt.legend()
    plt.savefig(dir+r"\\"+pic_name+"replication"+seconds+".jpg")
    plt.show()
    plt.close()
    #return
    t_e = np.std(mng_r-bmk_r)
    a_t_e = np.mean(np.abs(mng_r-bmk_r))
    rep_e = np.mean(np.abs(replication-hedged_v))
    return [t_e,a_t_e,rep_e]


def barra_optmize(mng_id,bmk_id =[],bmk_weight_tr = [],factor = factor,ras = [0.0075,0.0075]):
    id = mng_id + bmk_id
    #assign data, no need to change
    cov_data = cov_mx(factor)
    speRisk = Get_spec_risk(id)
    #speRisk_mng = Get_spec_risk(mng_id)
    expData = Get_Exposures(id,factor)
    #mng_expData = Get_Exposures(mng_id,factor)
    price = Get_price(id)
    #mng_price = Get_price(id)
    alpha = [0]*len(id)
    if len(bmk_weight_tr) > 0:
        # for active return type
        mngWeight = [1/len(mng_id)]*len(mng_id)+[0]*len(bmk_id)
        bmkWeight = bmk_weight_tr
        mngName = "managePortfolio"
        bmkName = "benchmarkPortfolio"
        #Lower bounds
        lB = [0]*len(mng_id)+[0]*len(bmk_id)
        #   Upper bounds
        uB = [1]*len(mng_id)+[0]*len(bmk_id)
    elif len(bmk_id)==0:
        mngWeight = [1 / len(id)] * len(id)
        bmkWeight = [0]*len(id)
        mngName = "managePortfolio"
        bmkName = "benchmarkPortfolio"
        # Lower bounds
        lB = [0.000]*(len(id)-1)+[0.01]
        #   Upper bounds
        uB = [0.5]*len(id)
    # Constants
    basevalue = 1000000.0
    cashflowweight = 0.0

    # Initialize java optimizer interface
    import barraopt

    # Create a workspace instance
    workspace = barraopt.CWorkSpace.CreateInstance()


    # Create a risk model
    rm = workspace.CreateRiskModel('SampleModel', barraopt.eEQUITY)

    # Add assets to workspace
    for i in range(len(id)):
        asset = workspace.CreateAsset(id[i], barraopt.eREGULAR)
        asset.SetAlpha(alpha[i])
        asset.SetPrice(price[i])

    # Set the covariance matrix from data object into the workspace
    for i in range(len(factor)):
        for j in range(i, len(factor)):
            rm.SetFactorCovariance(factor[i], factor[j], cov_data[i][j])
            # print(i,j)

    # Set the exposure matrix from data object
    for i in range(len(id)):
        for j in range(len(factor)):
            rm.SetFactorExposure(id[i], factor[j], expData[i][j])

    # Set the specific riks covariance matrix from data object
    for i in range(len(id)):
        rm.SetSpecificCovariance(id[i], id[i], speRisk[i])

    # Create managed, benchmark and universe portfolio
    mngPortfolio = workspace.CreatePortfolio(mngName)
    bmkPortfolio = workspace.CreatePortfolio(bmkName)
    uniPortfolio = workspace.CreatePortfolio('universePortfolio')

    # Set weights to portfolio assets
    for i in range(len(id)):
        mngPortfolio.AddAsset(id[i], mngWeight[i])
        bmkPortfolio.AddAsset(id[i], bmkWeight[i])
        uniPortfolio.AddAsset(id[i])  # no weight is needed for universe portfolio
    # Create a case
    testcase = workspace.CreateCase('SampleCase', mngPortfolio, uniPortfolio, basevalue, cashflowweight)
    # Initialize constraints
    constr = testcase.InitConstraints()
    linearConstr = constr.InitLinearConstraints()
    for i in range(len(id)):
        info = linearConstr.SetAssetRange(id[i])
        info.SetLowerBound(lB[i])
        info.SetUpperBound(uB[i])
    # Set Risk constraint
    riskconstr = constr.InitRiskConstraints()
    risk = riskconstr.AddPLTotalConstraint(True,bmkPortfolio)

    # Set risk model & term
    testcase.SetPrimaryRiskModel(workspace.GetRiskModel('SampleModel'));
    ut = testcase.InitUtility()

    ut.SetPrimaryRiskTerm(bmkPortfolio, ras[0], ras[1])

    solver = workspace.CreateSolver(testcase)

    # Uncomment the line below to dump workspace file
    # workspace.Serialize('opsdata.wsp')

    # Run optimizer
    status = solver.Optimize()

    #print the result of optimization
    if status.GetStatusCode() == barraopt.eOK:
        res = pd.DataFrame(columns=["value"])
        # Optimization completed
        output = solver.GetPortfolioOutput();
        risk = output.GetRisk();
        Spec_risk = output.GetSpecificRisk()
        utility = output.GetUtility();
        res.loc["total_risk"] = risk
        res.loc["utility"] = utility
        outputPortfolio = output.GetPortfolio();
        assetCount = outputPortfolio.GetAssetCount();

        outputWeight = [];
        for i in range(len(id)):
            outputWeight.append(outputPortfolio.GetAssetWeight(id[i]));

        # Show results on command window
        print('Optimization completed');
        print('Optimal portfolio risk: %g' % risk);
        print('Optimal portfolio Spec risk: %g' % Spec_risk);
        print('Optimal portfolio utility: %g' % utility);
        for i in range(len(id)):
            print('Optimal portfolio weight of asset %s: %g' % (id[i], outputWeight[i]));
            res.loc[id[i]] = outputWeight[i]
    else:
        # Optimization error
        print('Optimization error');
        sys.exit()

    workspace.Release();
    return(res)









'''#风格因子筛选，基于回归模型的R2的筛选
def factor_select(asset_ids,styles_exps,styles):
    ass = ",".join(asset_ids)
    rt_t1 = w.wsd(ass, "pct_chg","2021-11-01","2021-11-01",usedf=True)[1]
    exp_for_assets = styles_exps.copy()[exps["code"].isin(rt_t1.index)][["code","Factor","Exposure"]]
    data = pd.pivot_table(exp_for_assets,index = ["code"],columns = ["Factor"],values = ["Exposure"],fill_value=0)
    data.columns = [col[1] for col in data.columns]
    rt_t1.sort_index(ascending=True,inplace=True)
    data.sort_index(ascending=True,inplace = True)
    test_set = pd.merge(rt_t1,data,left_index=True,right_on="code")
    #ICs = [{i:test_set[["PCT_CHG",i]].corr(method="pearson").iloc[1,0]} for i in data.columns]
    vars_eval = pd.DataFrame(columns={"eval","vars"})
    indep_vars = []
    vars_list = []
    #向前逐步回归筛选因子
    for i in range(len(styles)):
        indep_vars.append([i])
    max_eval = 0
    best_eval = 0
    while(max_eval>=best_eval):
        vars_eval.drop(vars_eval.index, inplace=True)
        best_eval = max_eval
        for i in range(len(indep_vars)):
            for j in range(len(styles)):
                if j in indep_vars[i]:
                    continue
                vars = indep_vars[i].append(j)
                vars_list.append(vars)
                test = data.loc[:,vars]
                model = sm.OLS(endog=rt_t1,exog=test)
                est = model.fit()
                r2 = est.rsquared_adj
                vars_eval.loc[vars_eval.shape[0]] = [r2,vars]
        vars_eval.head(10)
        indep_vars.clear()
        indep_vars = vars_eval["vars"].tolist()
        max_eval = vars_eval["eval"].head(1).values[0]
        vars_eval.drop(vars_eval.index,inplace=True)

    return vars_eval
a = factor_select(bmk_id[:10],exps,factor)'''
