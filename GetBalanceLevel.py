from jqfactor import *
from jqdata import *
import pandas as pd
import CSVInit
import datetime
from math import isnan
import tushare as ts



def get_before_after_trade_days(date, count, is_before=True):
    """
    date :查询日期
    count : 前后追朔的数量
    is_before : True , 前count个交易日  ; False ,后count个交易日

    返回 : 基于date的日期, 向前或者向后count个交易日的日期 ,一个datetime.date 对象
    """
    all_date = pd.Series(get_all_trade_days())
    if isinstance(date,str):
        all_date = all_date.astype(str)
    if isinstance(date,datetime.datetime):
        date = date.date()

    if is_before :
        return all_date[all_date < date].tail(count).values[0]
    else :
        return all_date[all_date > date].head(count).values[-1]

    #获得当前股票当前日期的level，在start和end期间本只股票的最大增长率和大盘的最大增长率相减，即使实际增长率
def getLevel(securities, startDate, endDate, field,api):
    priceGap = get_price(securities, start_date=get_before_after_trade_days(startDate,1,False), end_date=endDate, fields = field)
    price = 0
    for pp in priceGap.iterrows():
        price = max(price, pp[1][0])
    init = get_price(securities, start_date=startDate, end_date=startDate, fields = field).iloc[0][0]
    priceSZ = 0
    dfSZ = get_price('000001.XSHG', start_date = str(get_before_after_trade_days(startDate, 1, False)), end_date =str(endDate))
    priceSZInit = get_price('000001.XSHG', start_date=str(startDate), end_date=str(startDate)).iloc[0]['high']
    for item in dfSZ.iterrows():
        priceSZ = max(priceSZ,item[1]['high'])
    rate = (price - init) / init
    if priceSZInit is None:
        rateSZ = 0
    else:
        rateSZ = (priceSZ - priceSZInit) / priceSZInit
    if (rate) < -0.3:
        return 0
    elif rate < 0:
        return 1
    elif rate < 0.3:
        return 2
    elif rate < 0.6:
        return 3
    else:
        return 4

#获得count内的收盘价均值，判断其是否当于date的收盘价
def getMAData(security, date, count):
    startDate = get_before_after_trade_days(date, count)
    prices = get_price(security, start_date = startDate, end_date = date, fields = ['close'])
    MA = 0
    cur = 0
    for price in prices.iterrows():
        MA += price[1][0]
        cur = price[1][0]
    MA /= count
    if(MA > cur):
        return 0
    else:
        return 1
    
    
ts.set_token('3ac5d90905e0e9bc413cfda768fff87ee0781a80702093b014c6cff4')
api = ts.pro_api()
times = 14  # 对于每一个茶柄形状的形成点要把包括前面的times天也放进随机森林
data = []  # 用来存储获得的字典列表以写入csv文件
factors = pd.read_csv("factor.csv")#读取存储因子名称得文件

excel = pd.read_excel(r"20210106.xlsx", usecols=[0, 2], names=None)
excel_li = excel.values.tolist()  # 按行获取excel数据并放进list
excel_li = excel_li[0 : (len(excel_li))]
number = 0
cur = datetime.datetime.now()
dataFiles = []

#数据量很大，所以分成多个数据文件
for i in range(0,51):
    dataFiles.append("data" + str(i))

    
initTime = datetime.datetime.now() #用来统计时间
for content in excel_li:  # 遍历每一只股票
    number += 1
    if number % 10 == 1:
        cur = datetime.datetime.now()
    print(number," ",str(content[0]))
    # 将股票代码转化为需要的格式
    securities = normalize_code(content[0])
    # 获取日期，并转换成需要的格式
    dateExcel = str(content[1])
    date =dateExcel[0:4] + '-' + dateExcel[4:6] + '-' + dateExcel[6:8]
    dicts = []
    for factor in factors:  # 遍历每一个因子
        rowData = get_factor_values(securities, factor, end_date=date, count=times)[factor] #获取每个因子count个日期的值，当非交易日时，获得的就是空数据
        index = 1 #每只股票都会存下多个日期的数据，index就是指现在正在操作哪一个日期
        for row in rowData.iterrows():
            if(len(dicts) < index):
                dicts.append({'securities':securities})
            dicts[index - 1]['date'] = datetime.datetime.strptime(str(row[0])[0:10],'%Y-%m-%d').date()
            if(isnan(row[1][0])): #如果因子值是nan，则自动将其化为0
                dicts[index - 1][factor] = 0
            else:
                dicts[index - 1][factor] = row[1][0]
            index += 1
    for dict in dicts:
        endDate = get_before_after_trade_days(dict['date'], times, False)
        dict['level'] = getLevel(securities, dict['date'], endDate, 'high',api)
        dict['MA_20'] = getMAData(securities,dict['date'],20)
        dict['MA_50'] = getMAData(securities, dict['date'], 50)
        data.append(dict)
    if (number % 10 == 0):
        print((number - 9), " ~ ", number, " 支股票生成数据文件耗时")
        m, s = divmod((datetime.datetime.now() - cur).seconds, 60)
        h, m = divmod(m, 60)
        print("%02d:%02d:%02d" % (h, m, s))
    if(number % 200 == 0):  #每操作两百只股票则生成一个文件   
        name = dataFiles[(int)(number / 200)]
        CSVInit.create_csv(name)
        CSVInit.writeDicts_csv(name, data)
        data = []
        print(" 前 ", number, " 支股票生成数据文件耗时")
        m, s = divmod((datetime.datetime.now() - initTime).seconds, 60)
        h, m = divmod(m, 60)
        print("%02d:%02d:%02d" % (h, m, s))
if(number % 200 != 0):    
    name = dataFiles[(int)(number / 200) + 1]
    CSVInit.create_csv(name)
    CSVInit.writeDicts_csv(name, data)
    print(" 前 ", number, " 支股票生成数据文件耗时")
    m, s = divmod((datetime.datetime.now() - initTime).seconds, 60)
    h, m = divmod(m, 60)
    print("%02d:%02d:%02d" % (h, m, s))
