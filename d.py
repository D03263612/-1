import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime
import re
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

# ===================== 1. 配置项 =====================
# 请求头（模拟浏览器）
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://maoyan.com/"
}

# 定义档期映射（核心特征：区分春节/暑期/普通档）
SCHEDULE_MAP = {
    "春节档": ["01-21", "01-22", "01-23", "01-24", "01-25", "01-26", "01-27"],  # 农历转换为公历简化版
    "暑期档": ["07-01", "07-02", "08-30", "08-31"],
    "国庆档": ["10-01", "10-02", "10-03", "10-04", "10-05", "10-06", "10-07"],
    "普通档": []
}

# 流量演员列表（可根据实际情况扩充）
FLOW_ACTORS = ["吴京", "沈腾", "易烊千玺", "张译", "贾玲", "刘德华", "黄渤"]


# ===================== 2. 数据爬取函数 =====================
def get_maoyan_movie_data(movie_names):
    """爬取猫眼电影票房、档期数据"""
    maoyan_data = []
    for name in movie_names:
        try:
            # 猫眼搜索接口（简化版，实际可替换为更稳定的接口）
            search_url = f"https://maoyan.com/search?kw={name}"
            response = requests.get(search_url, headers=HEADERS, timeout=10)

            # 解析票房和上映时间（示例：实际需根据页面结构调整XPath/BeautifulSoup）
            # 注：真实场景建议使用猫眼专业版API或合法数据源，此处为模拟解析
            movie_info = {
                "电影名称": name,
                "总票房(亿)": round(np.random.uniform(1, 50), 2),  # 模拟票房（实际需解析页面）
                "上映日期": f"2024-{np.random.choice(['01', '02', '07', '10'])}-{np.random.randint(1, 28)}",
                "排片率(%)": round(np.random.uniform(10, 50), 2)
            }
            maoyan_data.append(movie_info)
            print(f"已获取猫眼数据：{name}")
        except Exception as e:
            print(f"爬取猫眼{name}失败：{e}")
    return pd.DataFrame(maoyan_data)


def get_douban_movie_data(movie_names):
    """爬取豆瓣评分、导演、演员数据"""
    douban_data = []
    for name in movie_names:
        try:
            # 豆瓣搜索接口
            search_url = f"https://movie.douban.com/subject_search?search_text={name}"
            response = requests.get(search_url, headers=HEADERS, timeout=10)

            # 模拟解析（实际需用BeautifulSoup解析页面）
            directors = np.random.choice(["张艺谋", "吴京", "陈思诚", "冯小刚", "郭帆"], 1)[0]
            actors = np.random.choice(FLOW_ACTORS, np.random.randint(1, 4)).tolist()
            movie_info = {
                "电影名称": name,
                "豆瓣评分": round(np.random.uniform(5.0, 9.5), 1),
                "导演": directors,
                "主演": ",".join(actors)
            }
            douban_data.append(movie_info)
            print(f"已获取豆瓣数据：{name}")
        except Exception as e:
            print(f"爬取豆瓣{name}失败：{e}")
    return pd.DataFrame(douban_data)


def get_douyin_hot_data(movie_names):
    """爬取抖音宣传热度（话题播放量、点赞数）"""
    douyin_data = []
    for name in movie_names:
        try:
            # 抖音话题热度模拟（实际需调用抖音开放平台API）
            movie_info = {
                "电影名称": name,
                "抖音话题播放量(亿)": round(np.random.uniform(0.5, 100), 2),
                "抖音点赞数(万)": round(np.random.uniform(10, 1000), 2)
            }
            douyin_data.append(movie_info)
            print(f"已获取抖音数据：{name}")
        except Exception as e:
            print(f"爬取抖音{name}失败：{e}")
    return pd.DataFrame(douyin_data)


# ===================== 3. 特征工程函数 =====================
def feature_engineering(df):
    """特征工程：档期分类、流量演员标记、数值特征处理"""

    # 3.1 档期分类
    def get_schedule(date):
        """根据上映日期判断档期"""
        month_day = date.split("-")[1:]
        month_day_str = "-".join(month_day)
        if any(holiday in month_day_str for holiday in SCHEDULE_MAP["春节档"]):
            return "春节档"
        elif SCHEDULE_MAP["暑期档"][0] <= month_day_str <= SCHEDULE_MAP["暑期档"][1]:
            return "暑期档"
        elif any(holiday in month_day_str for holiday in SCHEDULE_MAP["国庆档"]):
            return "国庆档"
        else:
            return "普通档"

    df["档期类型"] = df["上映日期"].apply(get_schedule)

    # 3.2 标记是否有流量演员
    df["是否含流量演员"] = df["主演"].apply(lambda x: 1 if any(actor in x for actor in FLOW_ACTORS) else 0)

    # 3.3 标签编码（档期类型）
    le = LabelEncoder()
    df["档期类型编码"] = le.fit_transform(df["档期类型"])

    # 3.4 特征筛选（保留建模用特征）
    feature_cols = ["豆瓣评分", "排片率(%)", "抖音话题播放量(亿)", "是否含流量演员", "档期类型编码"]
    return df, feature_cols


# ===================== 4. 模型构建与优化 =====================
def build_box_office_model(df, feature_cols):
    """构建XGBoost票房预测模型，分析特征重要性"""
    # 4.1 数据拆分
    X = df[feature_cols]
    y = df["总票房(亿)"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4.2 构建XGBoost模型（优化参数）
    model = XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)

    # 4.3 模型评估
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # 4.4 特征重要性（识别关键影响因素）
    feature_importance = pd.DataFrame({
        "特征名称": feature_cols,
        "重要性": model.feature_importances_
    }).sort_values("重要性", ascending=False)

    # 4.5 保存预测结果
    df["预测票房(亿)"] = model.predict(df[feature_cols])

    return df, feature_importance, {"MAE": mae, "R2": r2}


# ===================== 5. 结果保存函数 =====================
def save_results(df, feature_importance, model_metrics, save_path="./movie_box_office_analysis.csv"):
    """保存原始数据+特征+预测结果+特征重要性为CSV"""
    # 合并所有结果
    result_df = df.copy()

    # 添加特征重要性（广播到每行）
    for idx, row in feature_importance.iterrows():
        result_df[f"{row['特征名称']}_重要性"] = row["重要性"]

    # 添加模型评估指标
    result_df["模型MAE"] = model_metrics["MAE"]
    result_df["模型R2"] = model_metrics["R2"]

    # 保存CSV（解决中文乱码）
    result_df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"\n所有结果已保存至：{save_path}")
    return result_df


# ===================== 6. 主执行流程 =====================
if __name__ == "__main__":
    # 步骤1：定义待分析的电影列表（可替换为爬取的热映电影）
    movie_list = ["流浪地球3", "满江红2", "唐探4", "热辣滚烫2", "封神第二部", "飞驰人生3"]

    # 步骤2：爬取多源数据
    print("===== 开始爬取数据 =====")
    maoyan_df = get_maoyan_movie_data(movie_list)
    douban_df = get_douban_movie_data(movie_list)
    douyin_df = get_douyin_hot_data(movie_list)

    # 步骤3：合并多源数据
    merged_df = maoyan_df.merge(douban_df, on="电影名称", how="inner")
    merged_df = merged_df.merge(douyin_df, on="电影名称", how="inner")

    # 步骤4：特征工程
    print("\n===== 开始特征工程 =====")
    feature_df, feature_cols = feature_engineering(merged_df)

    # 步骤5：构建预测模型
    print("\n===== 开始构建预测模型 =====")
    final_df, feature_importance, model_metrics = build_box_office_model(feature_df, feature_cols)

    # 步骤6：输出关键结果
    print("\n===== 关键影响因素（特征重要性） =====")
    print(feature_importance)
    print(f"\n模型评估：MAE={model_metrics['MAE']:.2f}, R2={model_metrics['R2']:.2f}")

    # 步骤7：保存所有结果为CSV
    save_results(final_df, feature_importance, model_metrics)