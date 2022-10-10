import sys
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from WindPy import w
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
def cov_mx(factor_ids,day): #factor_ids 因子名字符串列表
    #day = "20210901"
    local_id = pd.read_table(r"E:\research\exposure\CNE5S\CHN_LOCALID_Asset_ID."+day,sep = "|",header = 1).dropna(axis = 0).iloc[:,[0,2]]
    local_id["AssetID"] = [ code[2:]+".SH" if code[2:].startswith("6") else code[2:]+".SZ" for code in local_id["AssetID"]]
    global id_pair
    id_pair = local_id
    covs = pd.read_table(r"E:\research\exposure\CNE5S\CNE5S_100_Covariance."+day,sep = "|",header = 2)
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
def Get_spec_risk(asset_ids,day):
    asset_data = pd.read_table(r"E:\research\exposure\CNE5S\CNE5S_100_Asset_Data."+day,sep = "|",header = 2)
    asset_data = pd.merge(asset_data,id_pair)
    spec_risk = []
    for ids in asset_ids:
        spec_risk.append(asset_data.loc[asset_data["AssetID"]==ids,'SpecRisk%'].values[0])
    return spec_risk

#返回输入标的price
def Get_price(asset_ids,day):
    prices = pd.read_table(r"E:\research\exposure\CNE5S\CNE5_Daily_Asset_Price." + day, sep="|", header=1)
    prices = pd.merge(prices, id_pair)
    asset_price = []
    for ids in asset_ids:
        asset_price.append(prices.loc[prices["AssetID"] == ids, 'Price'].values[0])
    return asset_price
#返回标的因子exposure
def Get_Exposures(asset_ids,factor_ids,day):
    exps = pd.read_table(r"E:\research\exposure\CNE5S\CNE5S_100_Asset_Exposure." + day, sep="|", header=2)
    exps = pd.merge(exps, id_pair)
    asset_exp = []
    for i in asset_ids:
        spec_exp = []
        exp_p = exps[exps["AssetID"]==i]
        for j in factor_ids:
            try:
                spec_exp.append(exp_p.loc[exps['Factor']==j,'Exposure'].values[0])
            except:
                spec_exp.append(0) #industry factor
            #print(np.mat(asset_exp))
        asset_exp.append(spec_exp)
    return asset_exp

def track_error_cal(mng_id,bmk_id,mng_weight,bmk_weight=[],basevalue = 1000000,day = ""):
    if np.sum(mng_weight[:len(mng_id)]) < 0:
        bmk_weight = [0] * len(mng_id) + mng_weight[-len(bmk_id):]
        mng_weight = -1*mng_weight[:len(mng_id)] + [0] * len(bmk_id)
    id = mng_id + bmk_id
    code_str = ",".join(id)
    price = Get_price(id,day)
    mng_shares = np.mat([round(basevalue * mng_weight[i] / price[i]) for i in range(len(id))])
    bmk_shares = np.mat([round(basevalue * bmk_weight[i] / price[i]) for i in range(len(id))])
    price_series = w.wsd(code_str, "close", day, (dt.datetime.strptime(day,"%Y%m%d")+dt.timedelta(days=12)).strftime("%Y%m%d"), "", usedf=True)[1]
    price = np.mat(price_series.values)
    # value
    mng_v = price * mng_shares.T
    bmk_v = price * bmk_shares.T
    hedged_s = bmk_shares.T[-1]
    hedged_v = price[:, -1] * hedged_s
    replication = mng_v-bmk_v+hedged_v
    mng_r = mng_v / basevalue
    bmk_r = bmk_v / basevalue
    # return
    return [mng_r.getA1().tolist(),bmk_r.getA1().tolist(),hedged_v.getA1().tolist(),replication.getA1().tolist()]

def barra_optmize(mng_id,bmk_id =[],bmk_weight_tr = [],ras = [0.0075,0.0075],day = ""):

    covs = pd.read_table(r"E:\research\exposure\CNE5S\CNE5S_100_Covariance." + day, sep="|", header=2)
    factor = covs['!Factor1']
    factor = factor[~factor.duplicated()][:-1].tolist()

    id = mng_id + bmk_id
    #assign data, no need to change
    cov_data = cov_mx(factor,day)
    speRisk = Get_spec_risk(id,day)
    #speRisk_mng = Get_spec_risk(mng_id)
    expData = Get_Exposures(id,factor,day)
    #mng_expData = Get_Exposures(mng_id,factor)
    price = Get_price(id,day)
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


