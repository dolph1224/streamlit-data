import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import missingno as msno
import plotly.express as px

# 데이터 불러오기
df = pd.read_csv("data/diabetes_prediction_dataset.csv")
# st.dataframe(df)

# Step1. BMI의 이상치를 IQR(사분위수 범위) 방법을 사용하여 제거
# 결측치, 이상치 확인 및 처리
# 결측치 제거 (필요시)
# Step1. BMI의 이상치를 IQR(사분위수 범위) 방법을 사용하여 제거
Q1 = df['bmi'].quantile(0.25)  # 1사분위수 (25% 지점)
Q3 = df['bmi'].quantile(0.75)  # 3사분위수 (75% 지점)
IQR = Q3 - Q1  # 사분위범위

# 이상치 경계 계산
lower_bound = Q1 - 1.5 * IQR  # 이상치 하한값
upper_bound = Q3 + 1.5 * IQR  # 이상치 상한값

# BMI 이상치 제거
df = df[(df['bmi'] >= lower_bound) & (df['bmi'] <= upper_bound)]

# 결과 확인
# print(df['bmi'].describe())

# Step2. 데이터 타입 변환 및 정리
le_gender = LabelEncoder()
df["gender"] = le_gender.fit_transform(df["gender"])
le_smoking = LabelEncoder()
df["smoking_history"] = le_smoking.fit_transform(df["smoking_history"])
print(dict(zip(le_gender.classes_, le_gender.transform(le_gender.classes_))))
print(dict(zip(le_smoking.classes_, le_smoking.transform(le_smoking.classes_))))

# Step3: 특성과 타겟변수 분리
# 특성과 타겟 분리
# X (입력 데이터, Feature Set):
# gender (성별)
# age (나이)
# hypertension (고혈압)
# heart_disease (심장병)
# smoking_history (흡연 이력)
# bmi(체질량지수)
# HbA1c_level (당화혈색소 수치) 혈액 내 포도당과 혈색소가 결합한 수치, 당뇨병 환자의 혈당 관리 상태를 평가하는 항목으로 사용
# blood_glucose_level (혈액 속 포도당의 농도를 뜻하는 혈당 수치)

# y (타겟 데이터, diabetes):
# 당뇨 유무 (0,1)
# 이제 X와 y를 나눴으므로, 모델이 학습할 데이터를 준비한 상태입니다.
X=df[['gender','age','hypertension','heart_disease','smoking_history','bmi','HbA1c_level','blood_glucose_level']]
y=df['diabetes']

# Step4: 데이터 정규화(Scaling, 표준화)
# 머신러닝 모델은 특성(Feature) 값들의 크기가 다를 경우, 특정 값이 과도한 영향을 주는 문제(스케일링 문제)가 발생할 수 있음.
# 키(Height, cm)는 150~200 범위, 몸무게(Weight, kg)는 40~100 범위지만, BMI는 15~40 범위이므로 직접 비교가 어려움.
# StandardScaler()를 사용하여 모든 특성을 같은 범위(평균=0, 표준편차=1)로 변환하면 학습이 더 안정적임.
scaler = StandardScaler()
X_scaled =scaler.fit_transform(X)

# step5: 데이터셋 분할(train-test Split)
# 머신러닝 모델이 학습(Training) 과 평가(Testing) 를 위해 데이터를 분리해야 함.
# train_test_split() 함수를 사용하여 80%는 학습 데이터, 20%는 테스트 데이터로 분리합니다.
X_train, X_test, y_train, y_test = train_test_split(X_scaled,y,test_size=0.2,random_state=42)

# step6: 머신러닝 모델 학습
# 우리가 사용하는 모델은 랜덤 포레스트(Random Forest) 분류 모델입니다.
# 랜덤 포레스트는 결정 트리(Decision Tree)를 여러 개 조합하여 예측 정확도를 높이는 모델입니다
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 특성 중요도 출력
# 결과에 영향을 가장 많이 미치는 컬럼을 찾는 방법
# 모델이 예측하는 데 중요한 변수를 찾는 것이 핵심!
# 1) Feature Importance (특성 중요도) 확인
# 랜덤 포레스트 같은 트리 기반 모델을 활용하면, 어떤 변수가 모델 예측에 중요한 역할을 하는지 평가 가능.
feature_importances = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_})
feature_importances = feature_importances.sort_values(by="Importance", ascending=False)
print(feature_importances)

#Step7: 예측 및 성능평가(Model Prediction & Evaluation)
def classification_report_to_df(report):
    df_report = pd.DataFrame(report).transpose()
    return df_report
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

report_dict = classification_report(y_test, y_pred, target_names=np.unique(y_test).astype(str), output_dict=True)
classification_df = classification_report_to_df(report_dict)

# st.write(f'### Model Accuracy: {accuracy:.2f}')
# st.text(classification_df)

# Streamlit UI 디자인
st.set_page_config(page_title="Diabetes Dashboard", layout="wide")
# 사이드바 메뉴
st.markdown("""
    <style>
    div[data-baseweb="radio"] label div {
        visibility: hidden;
    }
    </style>
    """, unsafe_allow_html=True)

st.sidebar.title("📌 Navigation")

# 라디오 버튼 UI에서 선택 가능한 메뉴
menu = st.sidebar.selectbox(
    "메뉴 선택",
    ["🏠 Home", "📈 데이터 분석", "📉 데이터 시각화", "🤖 머신러닝 보고서"],
    index=0
)
st.header(menu)

# 홈화면
def home():
    st.title("당뇨병에 대한 개요")
    st.markdown("**데이터 출처** - [Kaggle](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset) Diabetes prediction dataset")
    st.subheader("당뇨병이란?")
    st.image('apple-8274593_640.jpg', width=500)
    st.markdown("""
    당뇨병(Diabetes Mellitus)은 혈당(혈액 중의 포도당) 수치가 비정상적으로 높아지는 만성 질환입니다. 이 질환은 인슐린 생산이 부족하거나 인슐린에 대한 신체의 반응이 저하되어 발생합니다. 인슐린은 췌장에서 분비되는 호르몬으로, 혈액 속의 포도당을 세포로 이동시켜 에너지원으로 사용되게 합니다.
    """)
    st.subheader("당뇨병의 종류")
    st.markdown("1. **제1형 당뇨병(Type 1 Diabetes)**: 주로 청소년기나 어린 시절에 발병하며, 면역체계가 췌장의 베타세포를 공격하여 인슐린 생산이 중단되는 자가면역 반응입니다.\n2. **제2형 당뇨병(Type 2 Diabetes)**: 성인기에 주로 발생하며, 인슐린 저항성이 주요 원인으로, 신체 세포가 인슐린에 제대로 반응하지 않게 됩니다.\n3. **임신성 당뇨병(Gestational Diabetes)**: 임신 중 발생할 수 있으며, 태반 호르몬의 영향으로 인슐린 저항성이 증가합니다.")
    st.subheader("당뇨병의 증상 및 합병증")
    st.markdown("- **주요 증상**: 다뇨(잦은 소변), 다갈(과도한 갈증), 다식(과도한 식욕), 체중 감소 등.\n- **합병증**: 장기적으로는 심혈관 질환, 신장 질환, 신경 손상, 시력 손상, 족부 궤양 등의 합병증을 초래할 수 있습니다.")
    st.subheader("관리 및 치료")
    st.markdown("- **식이 요법**: 균형 잡힌 식사 계획과 혈당 조절을 위한 식단 관리.\n- **운동**: 규칙적인 신체 활동으로 혈당 조절과 체중 관리.\n- **약물 치료**: 경구 혈당 강하제 또는 인슐린 주사.\n- **모니터링**: 정기적인 혈당 체크와 합병증 예방을 위한 검사.")

# 데이터 분석
def analyze_data():
    st.subheader("당뇨병 예측 데이터셋 개요")
    st.markdown("""
        이 데이터셋은 당뇨병 예측을 위한 의료 데이터를 포함하고 있으며, 총 **100,000개의 샘플**과 **9개의 주요 특성**으로 구성되어 있습니다.
        데이터는 환자의 건강 상태, 생활 습관 및 혈당 수치와 관련된 정보를 포함하여 당뇨병 예측 모델 개발에 유용하게 활용될 수 있습니다.
    """)
    col1, col2 = st.columns(2)
    col1.markdown("""
        ### **데이터셋 특징**
        1. **성별 (gender)**: 환자의 성별 정보 (Male/Female)
        2. **나이 (age)**: 환자의 연령 (최소 0.08세 ~ 최대 80세)
        3. **고혈압 (hypertension)**: 고혈압 여부 (0: 없음, 1: 있음)
        4. **심장병 (heart_disease)**: 심장병 여부 (0: 없음, 1: 있음)
        5. **흡연 이력 (smoking_history)**: 환자의 흡연 상태 (never, current, former, No Info 등)
        6. **체질량지수 (bmi)**: BMI 수치 (10.01 ~ 95.69)
        7. **혈당 조절 지표 (HbA1c_level)**: 장기적인 혈당 조절 상태를 나타내는 수치 (3.5 ~ 9.0)
        8. **혈당 수치 (blood_glucose_level)**: 혈당 검사 결과 (80 ~ 300)
        9. **당뇨병 여부 (diabetes)**: 당뇨병 진단 결과 (0: 없음, 1: 있음)
    """)
    col2.markdown("""
        ### **주요 통계 정보**
        - **평균 나이**: 약 41.89세
        - **평균 BMI**: 27.32
        - **당뇨병 환자 비율**: 8.5% (전체의 8,500명)
        - **고혈압 환자 비율**: 7.5%
        - **심장병 환자 비율**: 3.9%
    """)

    col1, col2 = st.columns(2)    
    # 독립 변수 설명
    col1.subheader("독립 변수 (Feature Set)")
    col1.markdown("""
    - **gender**: 성별 (남성 = 0, 여성 = 1)
    - **age**: 나이 (단위: 년)
    - **hypertension**: 고혈압 여부 (없음 = 0, 있음 = 1)
    - **heart_disease**: 심장병 여부 (없음 = 0, 있음 = 1)
    - **smoking_history**: 흡연 이력 (비흡연자 = 0, 흡연자 = 1)
    - **bmi**: 체질량지수 (kg/m²)
    - **HbA1c_level**: 당화혈색소 수치 (%)
    - **blood_glucose_level**: 공복 혈당 수치 (mg/dL)
    """)
    
    # 종속 변수 설명
    col2.subheader("종속 변수 (Target Variable)")
    col2.markdown(" - **diabetes**: 당뇨병 유무 (당뇨 없음 = 0, 당뇨 있음 = 1)")

    # 데이터 통계 정보
    st.write("데이터 통계 요약")
    st.write(df.describe())
    
    # 데이터에서 당뇨병 유무 분포 시각화
    st.subheader("당뇨병 유무 분포")
    df['diabetes'] = df['diabetes'].map({1: '있음', 0: '없음'})
    fig = px.histogram(df, x="diabetes", color="diabetes", title="당뇨병 유무 분포", 
                    color_discrete_sequence=["#FF7F0E", "#1F77B4"])
    fig.update_layout(bargap=0.2)  # 막대 간 간격 설정
    st.plotly_chart(fig)
    
    # 결측치 시각화
    st.subheader("결측치 시각화")
    fig = plt.figure(figsize=(10, 4))
    msno.bar(df, color="gray")
    st.pyplot(fig)



# EDA(데이터 시각화 화면)
def eda():
    st.title("데이터 시각화")
    chart_tabs = st.tabs(['histogram','boxplot','hitmap'])
    gender_mapping = {0: 'Female', 1: 'Male', 2: 'Other'}
    df['gender'] = df['gender'].map(gender_mapping)
    with chart_tabs[0]:
        st.subheader("연령, 흡연이력, 당화혈색소 수치, BMI 분포")
        fig, axes = plt.subplots(2,2, figsize=(12,8))
        columns = ["age","smoking_history","HbA1c_level","bmi"]
        for i, col in enumerate(columns):
            ax = axes[i//2, i%2]
            sns.histplot(df[col],bins=20,kde=True, ax=ax) 
            #bins=20 20개 구간 만큼의 빈도수를 알려줌 kde->데이터의 분포를 부드러운 곡선으로 확인
            ax.set_title(col)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)  # wspace: 가로 간격, hspace: 세로 간격
        # plt.tight_layout()
        st.pyplot(fig)
    # 박스플롯
    with chart_tabs[1]:
        st.subheader("성별 및 당화혈색소 수치 박스플롯")
        fig, ax = plt.subplots(figsize=(8,5))
        sns.boxplot(data=df, x="gender", y="HbA1c_level", hue="diabetes", palette="Set3",ax=ax)
        ax.set_title("HbA1c level-BOXPLOT")
        st.pyplot(fig)
    # 변수간 상관관계 분석 (히트맵)
    with chart_tabs[2]:
        st.subheader("상관관계 히트맵")
        fig,ax = plt.subplots(figsize=(8,6))
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f",linewidths=0.5, ax=ax)
        ax.set_title("Feature Correlation Hitmap")
        st.pyplot(fig)

def plot_feature_importance_plotly(importance_df):
    fig = px.bar(importance_df, 
                 x="Importance", y="Feature", 
                 title="Feature Importance",
                 labels={"Importance": "Importance", "Feature": "Feature"},
                 color="Importance",  # 색상 설정
                 color_continuous_scale="Viridis")  # 색상 팔레트

    fig.update_layout(
        xaxis_title="Importance",
        yaxis_title="Feature",
        template="plotly_dark",  # 어두운 테마로 설정
        showlegend=False
    )

    st.plotly_chart(fig)  # Streamlit에서 Plotly 차트 출력

# 머신러닝 보고서 함수 정의
def ml_performance_report():
    st.subheader("모델 성능 평가")
    st.write("### 모델 정확도: {:.2f}".format(accuracy))
    st.subheader("📌 분류 보고서")
    st.write(classification_df)
    
    st.subheader("📌 특성 중요도 분석")
    feature_importances = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_})
    feature_importances = feature_importances.sort_values(by="Importance", ascending=False)
    st.write(feature_importances)
    
    # fig, ax = plt.subplots()
    # sns.barplot(x=feature_importances["Feature"], y=feature_importances["Importance"], ax=ax)
    # st.pyplot(fig)
    
    # Plotly로 바 차트 그리기
    fig = px.bar(feature_importances, x="Feature", y="Importance", title="Feature Importances")
    # Streamlit에서 Plotly 그래프 표시
    st.plotly_chart(fig)


    

# 메뉴 선택에 따른 화면 전환
if menu == "🏠 Home":
    home()
elif menu == "📈 데이터 분석":
    analyze_data()
elif menu == "📉 데이터 시각화":
    eda()
elif menu == "🤖 머신러닝 보고서":
    ml_performance_report()
