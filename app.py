import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import os
import datetime

# ─────────────────────────────────────────────────────────
# 🟢 1. 기본 설정
# ─────────────────────────────────────────────────────────
st.set_page_config(page_title="MM 월간 공급량 예측 (당월 추정)", layout="wide")

def set_korean_font():
    try:
        import matplotlib as mpl
        mpl.rcParams['axes.unicode_minus'] = False
        mpl.rc('font', family='Malgun Gothic') 
    except: pass

set_korean_font()

# ─────────────────────────────────────────────────────────
# 🟢 2. 데이터 로드 및 전처리 함수
# ─────────────────────────────────────────────────────────
@st.cache_data
def load_raw_data(uploaded_file):
    """과거 실적 데이터 로드"""
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('.xlsx') else pd.read_csv(uploaded_file)
        except:
            df = pd.read_csv(uploaded_file, encoding='cp949')
    elif os.path.exists("일일공급량_raw.xlsx"):
        df = pd.read_excel("일일공급량_raw.xlsx")
    else:
        return pd.DataFrame() # 파일 없음

    # 컬럼 공백 제거
    df.columns = df.columns.str.strip()
    
    # 날짜 변환
    if '일자' in df.columns:
        df['일자'] = pd.to_datetime(df['일자'])
        df['연'] = df['일자'].dt.year
        df['월'] = df['일자'].dt.month
        df['일'] = df['일자'].dt.day
    
    # MJ 단위 처리 (혹시 콤마가 문자열로 들어가있을 경우 대비)
    if '공급량(MJ)' in df.columns and df['공급량(MJ)'].dtype == object:
        df['공급량(MJ)'] = df['공급량(MJ)'].astype(str).str.replace(',', '').astype(float)

    return df

def create_forecast_template(year, month):
    """예측용 템플릿 생성 (해당 월의 모든 날짜)"""
    import calendar
    _, last_day = calendar.monthrange(year, month)
    dates = [datetime.date(year, month, day) for day in range(1, last_day + 1)]
    return pd.DataFrame({'일자': pd.to_datetime(dates)})

# ─────────────────────────────────────────────────────────
# 🟢 3. 머신러닝 모델 학습
# ─────────────────────────────────────────────────────────
def train_avg_temp_model(df):
    """1. 최저/최고 기온으로 평균기온을 맞추는 모델"""
    df_clean = df.dropna(subset=['최저기온(℃)', '최고기온(℃)', '평균기온(℃)'])
    X = df_clean[['최저기온(℃)', '최고기온(℃)']]
    y = df_clean['평균기온(℃)']
    
    model = LinearRegression()
    model.fit(X, y)
    return model

def train_supply_model(df):
    """2. 평균기온으로 공급량(MJ)을 맞추는 모델 (2차 곡선 회귀)"""
    # 동절기 예측 정확도를 위해 전체 데이터 사용 (또는 10~4월만 필터링 가능)
    df_clean = df.dropna(subset=['평균기온(℃)', '공급량(MJ)'])
    # 0인 데이터 제외 (이상치)
    df_clean = df_clean[df_clean['공급량(MJ)'] > 0]
    
    X = df_clean[['평균기온(℃)']]
    y = df_clean['공급량(MJ)']
    
    # 기온과 가스는 비선형 관계(추울수록 급격히 증가)이므로 2차 다항회귀 사용
    model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    model.fit(X, y)
    return model

# ─────────────────────────────────────────────────────────
# 🟢 4. 메인 로직
# ─────────────────────────────────────────────────────────
def main():
    st.title("📊 MM 회의용 월간 공급량 예측 (2월)")
    st.markdown("#### 💡 기상청 예보(최저/최고) 기반 당월 실적 추정 시스템")

    # 1. 사이드바 - 파일 업로드
    with st.sidebar:
        st.header("📂 데이터 파일 관리")
        
        # A. 과거 실적 파일 (기본 파일 자동 로드 시도)
        up_raw = st.file_uploader("1. 과거 실적 데이터 (일일공급량_raw.xlsx)", type=['xlsx', 'csv'])
        
        df_raw = load_raw_data(up_raw)
        
        if df_raw.empty:
            st.error("⚠️ '일일공급량_raw.xlsx' 파일을 업로드하거나 폴더에 넣어주세요.")
            return

        st.success(f"✅ 과거 데이터 로드 완료: {len(df_raw):,}건")
        
        st.markdown("---")
        
        # B. 예측 대상 월 설정
        st.subheader("📅 예측 대상 설정")
        # 현재 날짜 기준 다음달 자동 세팅 (예: 지금 2026-02-09면 2월)
        today = datetime.date.today()
        target_year = st.number_input("연도 (Year)", value=2026)
        target_month = st.number_input("월 (Month)", value=2)
        
        st.markdown("---")
        
        # C. 향후 예보 입력 방식
        st.subheader("🌡️ 향후 기온 예보 입력")
        input_method = st.radio("입력 방식", ["직접 입력 (표)", "엑셀 업로드"], index=0)
        
        forecast_input = None
        if input_method == "엑셀 업로드":
            up_forecast = st.file_uploader("2. 기상청 예보 파일 (최저/최고)", type=['xlsx', 'csv'])
            if up_forecast:
                forecast_input = pd.read_excel(up_forecast) if up_forecast.name.endswith('.xlsx') else pd.read_csv(up_forecast)

    # 2. 본문 - 모델 학습
    model_temp = train_avg_temp_model(df_raw)
    model_supply = train_supply_model(df_raw)

    # 3. 당월 데이터 프레임 생성 (1일 ~ 말일)
    df_current_month = create_forecast_template(target_year, target_month)
    
    # 4. 기존 실적 매핑 (이미 지나간 날짜)
    mask_past = (df_raw['연'] == target_year) & (df_raw['월'] == target_month)
    df_actuals = df_raw[mask_past][['일자', '평균기온(℃)', '최저기온(℃)', '최고기온(℃)', '공급량(MJ)']]
    
    # 병합: 실적 있으면 실적 사용, 없으면 NaN
    df_merged = pd.merge(df_current_month, df_actuals, on='일자', how='left')
    
    # 5. 미래 구간 구분
    # 실적(공급량)이 없는 날짜를 미래로 간주
    missing_indices = df_merged[df_merged['공급량(MJ)'].isna()].index
    
    if len(missing_indices) == 0:
        st.success("✅ 해당 월의 실적이 모두 확정되었습니다.")
        st.dataframe(df_merged)
        return

    st.info(f"📌 현재 **{len(missing_indices)}일** 간의 실적이 비어 있습니다. 예측을 시작합니다.")

    # 6. 사용자 기온 입력 (미래 10~13일치)
    st.markdown("### 1️⃣ 향후 기온 정보 입력 (기상청 예보)")
    
    # 편집 가능한 데이터프레임 준비 (미래 구간만)
    df_future_input = df_merged.loc[missing_indices, ['일자', '최저기온(℃)', '최고기온(℃)']].copy()
    
    if forecast_input is not None:
        # 업로드된 파일이 있으면 병합 시도 (날짜 기준)
        # (구현 생략: 간단히 직접 입력 권장 또는 인덱스 매칭)
        pass
        
    # 데이터 에디터 출력
    edited_temps = st.data_editor(
        df_future_input, 
        num_rows="fixed", 
        hide_index=True,
        column_config={
            "일자": st.column_config.DateColumn("날짜", format="YYYY-MM-DD", disabled=True),
            "최저기온(℃)": st.column_config.NumberColumn("최저기온 (기상청)", required=True),
            "최고기온(℃)": st.column_config.NumberColumn("최고기온 (기상청)", required=True),
        }
    )
    
    # 7. 빈 구간(기상청 예보도 없는 먼 미래) 채우기 전략
    st.markdown("### 2️⃣ 예보가 없는 구간(먼 미래) 추정 방식")
    fill_strategy = st.radio(
        "기상청 예보조차 없는 날짜의 기온은 어떻게 채울까요?",
        ["과거 3년 동월 평균 기온 적용", "전년도(작년) 동일 날짜 기온 적용"],
        horizontal=True
    )
    
    # 8. 최종 예측 실행 버튼
    if st.button("🚀 월간 공급량 예측 실행 (Click)", type="primary"):
        df_final = df_merged.copy()
        
        # (1) 사용자 입력값(기상청 예보) 반영
        df_final.set_index('일자', inplace=True)
        edited_temps.set_index('일자', inplace=True)
        df_final.update(edited_temps) # 인덱스 기준으로 업데이트
        df_final.reset_index(inplace=True)
        
        # (2) 아직도 비어있는 기온(먼 미래) 채우기
        null_temp_indices = df_final[df_final['최저기온(℃)'].isna()].index
        
        for idx in null_temp_indices:
            target_date = df_final.loc[idx, '일자']
            md_month, md_day = target_date.month, target_date.day
            
            if "과거 3년" in fill_strategy:
                # 과거 3년치 동일 날짜 필터링
                past_years = [target_year-1, target_year-2, target_year-3]
                past_data = df_raw[
                    (df_raw['연'].isin(past_years)) & 
                    (df_raw['월'] == md_month) & 
                    (df_raw['일'] == md_day)
                ]
                # 없으면 월평균으로 대체
                if past_data.empty:
                    past_data = df_raw[(df_raw['연'].isin(past_years)) & (df_raw['월'] == md_month)]
                
                fill_min = past_data['최저기온(℃)'].mean()
                fill_max = past_data['최고기온(℃)'].mean()
                
            else: # 전년도 동일 날짜
                past_data = df_raw[
                    (df_raw['연'] == target_year-1) & 
                    (df_raw['월'] == md_month) & 
                    (df_raw['일'] == md_day)
                ]
                if past_data.empty: # 작년 데이터 없으면 재작년
                     past_data = df_raw[(df_raw['연'] == target_year-2) & (df_raw['월'] == md_month) & (df_raw['일'] == md_day)]
                
                if not past_data.empty:
                    fill_min = past_data['최저기온(℃)'].values[0]
                    fill_max = past_data['최고기온(℃)'].values[0]
                else:
                    fill_min, fill_max = 0, 10 # 기본값 (예외처리)

            df_final.loc[idx, '최저기온(℃)'] = fill_min
            df_final.loc[idx, '최고기온(℃)'] = fill_max
            df_final.loc[idx, '비고'] = "추세추정"

        # (3) 평균 기온 추정 (AI 모델 1: Min/Max -> Avg)
        # 평균기온이 비어있는 행만 대상
        mask_avg_null = df_final['평균기온(℃)'].isna()
        if mask_avg_null.sum() > 0:
            X_pred = df_final.loc[mask_avg_null, ['최저기온(℃)', '최고기온(℃)']]
            predicted_avg = model_temp.predict(X_pred)
            df_final.loc[mask_avg_null, '평균기온(℃)'] = predicted_avg

        # (4) 공급량 추정 (AI 모델 2: Avg -> Supply)
        # 공급량이 비어있는 행만 대상
        mask_supply_null = df_final['공급량(MJ)'].isna()
        if mask_supply_null.sum() > 0:
            X_supply = df_final.loc[mask_supply_null, ['평균기온(℃)']]
            predicted_supply = model_supply.predict(X_supply)
            df_final.loc[mask_supply_null, '공급량(MJ)'] = predicted_supply
            df_final.loc[mask_supply_null, '구분'] = '예측'
        
        df_final['구분'] = df_final['구분'].fillna('실적')

        # 9. 결과 시각화
        st.divider()
        st.subheader(f"📈 {target_month}월 공급량 예측 결과")
        
        total_supply = df_final['공급량(MJ)'].sum()
        current_sum = df_final[df_final['구분']=='실적']['공급량(MJ)'].sum()
        pred_sum = df_final[df_final['구분']=='예측']['공급량(MJ)'].sum()
        
        # KPI 카드
        c1, c2, c3 = st.columns(3)
        c1.metric("총 예상 공급량 (MJ)", f"{total_supply:,.0f}")
        c2.metric("현재 마감 실적 (MJ)", f"{current_sum:,.0f}")
        c3.metric("남은 기간 예측 (MJ)", f"{pred_sum:,.0f}")
        
        # 그래프 (콤보 차트: 공급량 막대 + 기온 꺾은선)
        fig = go.Figure()
        
        # 공급량 (막대)
        fig.add_trace(go.Bar(
            x=df_final['일자'], 
            y=df_final['공급량(MJ)'], 
            name='공급량(MJ)',
            marker_color=df_final['구분'].map({'실적': 'navy', '예측': 'orange'}),
            opacity=0.7
        ))
        
        # 기온 (선) - 이중축
        fig.add_trace(go.Scatter(
            x=df_final['일자'], 
            y=df_final['평균기온(℃)'], 
            name='평균기온',
            mode='lines+markers',
            line=dict(color='red', width=2),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title=f"{target_year}년 {target_month}월 일별 공급량 및 기온 예측",
            yaxis=dict(title="공급량 (MJ)"),
            yaxis2=dict(title="평균기온 (℃)", overlaying='y', side='right'),
            legend=dict(x=0, y=1.1, orientation='h'),
            xaxis=dict(tickformat="%d일", dtick="D1")
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 10. 상세 데이터 표 및 다운로드
        with st.expander("📋 일별 상세 데이터 확인"):
            # 소수점 정리
            df_display = df_final.copy()
            df_display['공급량(MJ)'] = df_display['공급량(MJ)'].round(0)
            df_display['평균기온(℃)'] = df_display['평균기온(℃)'].round(1)
            
            st.dataframe(df_display, use_container_width=True)
            
            # CSV 다운로드
            csv = df_display.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                "📥 예측 결과 다운로드 (CSV)",
                csv,
                f"{target_year}년_{target_month}월_MM예측자료.csv",
                "text/csv"
            )

if __name__ == "__main__":
    main()
