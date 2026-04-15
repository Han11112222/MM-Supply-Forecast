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
import calendar

# ─────────────────────────────────────────────────────────
# 🟢 1. 기본 설정
# ─────────────────────────────────────────────────────────
st.set_page_config(page_title="MM Supply Forecast", layout="wide")

def set_korean_font():
    try:
        import matplotlib as mpl
        mpl.rcParams['axes.unicode_minus'] = False
        mpl.rc('font', family='Malgun Gothic') 
    except: pass

set_korean_font()

# ─────────────────────────────────────────────────────────
# 🟢 2. 데이터 로드 및 모델링
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
        return pd.DataFrame()

    df.columns = df.columns.str.strip()
    if '일자' in df.columns:
        df['일자'] = pd.to_datetime(df['일자'])
        df['연'] = df['일자'].dt.year
        df['월'] = df['일자'].dt.month
        df['일'] = df['일자'].dt.day
     
    if '공급량(MJ)' in df.columns and df['공급량(MJ)'].dtype == object:
        df['공급량(MJ)'] = df['공급량(MJ)'].astype(str).str.replace(',', '').astype(float)

    return df

def train_models(df):
    """
    Model 1: 최저/최고 기온 -> 평균 기온 (선형회귀)
    Model 2: 평균 기온 -> 공급량 (3차 다항회귀)
    """
    # 1. Temp Model
    df_t = df.dropna(subset=['최저기온(℃)', '최고기온(℃)', '평균기온(℃)'])
    model_temp = LinearRegression()
    model_temp.fit(df_t[['최저기온(℃)', '최고기온(℃)']], df_t['평균기온(℃)'])
     
    # 2. Supply Model
    df_s = df.dropna(subset=['평균기온(℃)', '공급량(MJ)'])
    df_s = df_s[df_s['공급량(MJ)'] > 0]
     
    # 🟢 수정: 2차 -> 3차 다항회귀로 변경 (include_bias=False로 수식 추출 용이하게 설정)
    model_supply = make_pipeline(PolynomialFeatures(degree=3, include_bias=False), LinearRegression())
    model_supply.fit(df_s[['평균기온(℃)']], df_s['공급량(MJ)'])
    
    # 🟢 추가: 3차 다항식 수식 추출
    reg = model_supply.named_steps['linearregression']
    c0 = reg.intercept_
    c1, c2, c3 = reg.coef_
    formula = f"y = ({c3:.4e})x³ + ({c2:.4e})x² + ({c1:.4e})x + {c0:,.0f}"
     
    return model_temp, model_supply, formula

def get_past_stats(df_raw, target_month, method="3년 평균"):
    """과거 데이터 통계 추출 (빈 날짜 채우기용)"""
    stats_dict = {} 
     
    df_past = df_raw[df_raw['월'] == target_month].copy()
    if df_past.empty: return stats_dict
     
    max_year = df_past['연'].max()
    if method == "3년 평균":
        target_years = [max_year-1, max_year-2, max_year-3]
    else: # 전년도
        target_years = [max_year-1]
         
    df_past = df_past[df_past['연'].isin(target_years)]
     
    grp = df_past.groupby('일')[['최저기온(℃)', '최고기온(℃)']].mean()
     
    for day, row in grp.iterrows():
        stats_dict[(target_month, day)] = (row['최저기온(℃)'], row['최고기온(℃)'])
         
    return stats_dict

# ─────────────────────────────────────────────────────────
# 🟢 3. 메인 로직
# ─────────────────────────────────────────────────────────
def main():
    st.title("📊 MM Supply Forecast (당월 마감 및 예측)")
     
    # 1. 사이드바
    with st.sidebar:
        st.header("📂 데이터 및 설정")
        up_raw = st.file_uploader("1. 과거 실적 (일일공급량_raw.xlsx)", type=['xlsx', 'csv'])
        df_raw = load_raw_data(up_raw)
         
        if df_raw.empty:
            st.error("⚠️ 파일을 업로드하거나 폴더에 넣어주세요.")
            return

        # 🟢 수정 1: 학습 데이터 연도 선택 기능 추가
        st.markdown("---")
        st.subheader("🎯 학습 데이터 범위 설정")
        available_years = sorted(df_raw['연'].unique().tolist(), reverse=True)
        # 기본값으로 최근 3년 설정
        default_years = available_years[:min(3, len(available_years))]
        
        selected_years = st.multiselect(
            "학습에 사용할 연도 선택",
            options=available_years,
            default=default_years,
            help="특정 연도(예: 2021년)를 제외하고 예측 모델을 학습시킬 수 있습니다."
        )
        
        # 선택된 연도의 데이터만 추출하여 학습에 사용
        df_train = df_raw[df_raw['연'].isin(selected_years)]
        
        if df_train.empty:
            st.warning("⚠️ 선택된 연도가 없습니다. 학습할 연도를 선택해주세요.")
            return
         
        st.markdown("---")
        st.subheader("📅 마감 대상 월 설정")
        today = datetime.date.today()
        target_year = st.number_input("연도 (Year)", value=today.year)
        target_month = st.number_input("월 (Month)", value=today.month)
         
        st.markdown("---")
        st.subheader("⚙️ 추정 옵션")
        fill_method = st.radio("미입력 구간(먼 미래) 기온 대체 방식", ["과거 3년 평균", "전년도 실적"])

    # 2. 모델 학습 (필터링된 df_train 사용)
    model_temp, model_supply, formula_str = train_models(df_train)

    # 3. 데이터 프레임 생성
    _, last_day = calendar.monthrange(target_year, target_month)
    dates = [datetime.date(target_year, target_month, d) for d in range(1, last_day + 1)]
    df_curr = pd.DataFrame({'일자': pd.to_datetime(dates)})
     
    # 4. 실적 매핑
    mask_month = (df_raw['연'] == target_year) & (df_raw['월'] == target_month)
    df_actual = df_raw[mask_month][['일자', '공급량(MJ)', '평균기온(℃)', '최저기온(℃)', '최고기온(℃)']]
     
    df_merged = pd.merge(df_curr, df_actual, on='일자', how='left')
    df_merged['구분'] = np.where(df_merged['공급량(MJ)'].notnull(), '실적', '예측대상')
     
    # 5. 사용자 입력
    missing_idx = df_merged[df_merged['구분'] == '예측대상'].index
     
    if len(missing_idx) > 0:
        st.info(f"📌 **{target_month}월**의 남은 **{len(missing_idx)}일**에 대한 예측을 수행합니다. (학습 연도: {', '.join(map(str, selected_years))})")
         
        df_input = df_merged.loc[missing_idx, ['일자', '최저기온(℃)', '최고기온(℃)']].copy()
        
        st.markdown("### 1️⃣ 기상청 예보 입력 (최저/최고)")
        st.caption("👇 아래 표를 수정하면 그래프에 실시간으로 반영됩니다.")
            
        edited_df = st.data_editor(
            df_input,
            num_rows="fixed", hide_index=True,
            column_config={
                "일자": st.column_config.DateColumn("날짜", format="MM-DD", disabled=True),
                "최저기온(℃)": st.column_config.NumberColumn("최저기온", required=True),
                "최고기온(℃)": st.column_config.NumberColumn("최고기온", required=True),
            },
            use_container_width=True
        )
        
        st.markdown("---") 
        st.markdown("### 2️⃣ 분석 실행")
        st.write("입력된 예보와 선택된 과거 데이터를 결합하여 최종 공급량을 추정합니다.")
        run_btn = st.button("🚀 예측 실행 및 그래프 그리기", type="primary", use_container_width=True) 
             
        if run_btn:
            # A. 데이터 업데이트
            df_final = df_merged.copy()
            for idx in edited_df.index:
                t_min, t_max = edited_df.loc[idx, '최저기온(℃)'], edited_df.loc[idx, '최고기온(℃)']
                df_final.loc[idx, '최저기온(℃)'], df_final.loc[idx, '최고기온(℃)'] = t_min, t_max
                df_final.loc[idx, '데이터출처'] = '예보(입력)' if pd.notnull(t_min) and pd.notnull(t_max) else '과거패턴'

            # B. 빈값 채우기 (기존 df_raw 전체를 기온 패턴용으로 사용)
            stats_map = get_past_stats(df_raw, target_month, fill_method)
            for i, row in df_final.iterrows():
                if pd.isnull(row['최저기온(℃)']) or pd.isnull(row['최고기온(℃)']):
                    md = (row['일자'].month, row['일자'].day)
                    if md in stats_map:
                        df_final.at[i, '최저기온(℃)'], df_final.at[i, '최고기온(℃)'] = stats_map[md]
                        df_final.at[i, '데이터출처'] = '과거패턴'
             
            # C. 평균기온 추정
            mask_avg = df_final['평균기온(℃)'].isna()
            if mask_avg.sum() > 0:
                X_pred = df_final.loc[mask_avg, ['최저기온(℃)', '최고기온(℃)']].fillna(0)
                df_final.loc[mask_avg, '평균기온(℃)'] = model_temp.predict(X_pred)
             
            # D. 공급량 추정
            mask_supply = df_final['공급량(MJ)'].isna()
            if mask_supply.sum() > 0:
                X_supply = df_final.loc[mask_supply, ['평균기온(℃)']]
                df_final.loc[mask_supply, '공급량(MJ)'] = model_supply.predict(X_supply)
             
            # E. 실적 마킹
            df_final['데이터출처'] = df_final['데이터출처'].fillna('실적')
             
            # 6. 결과 시각화
            st.divider()
            st.subheader(f"📈 {target_year}년 {target_month}월 공급량 예측 결과")
             
            total_sum = df_final['공급량(MJ)'].sum()
            closed_sum = df_final[df_final['데이터출처']=='실적']['공급량(MJ)'].sum()
            forecast_sum = total_sum - closed_sum
             
            k1, k2, k3 = st.columns(3)
            k1.metric("총 예상 공급량", f"{total_sum/1000:,.0f} GJ", "당월 합계")
            k2.metric("마감 실적", f"{closed_sum/1000:,.0f} GJ", "확정분")
            k3.metric("예측 잔여량", f"{forecast_sum/1000:,.0f} GJ", "추정분")
             
            fig = go.Figure()
            color_map = {'실적': '#1f77b4', '예보(입력)': '#ff7f0e', '과거패턴': '#7f7f7f'}
            for source in ['실적', '예보(입력)', '과거패턴']:
                df_sub = df_final[df_final['데이터출처'] == source]
                if not df_sub.empty:
                    fig.add_trace(go.Bar(x=df_sub['일자'], y=df_sub['공급량(MJ)'], name=f"{source}", marker_color=color_map[source]))

            fig.add_trace(go.Scatter(x=df_final['일자'], y=df_final['평균기온(℃)'], name='평균기온(추정)', mode='lines+markers', line=dict(color='red', width=2, dash='dot'), yaxis='y2'))
             
            fig.update_layout(
                title=dict(text=f"일별 공급량 및 기온 추이 ({target_month}월)", font=dict(size=20)),
                yaxis=dict(title="공급량 (MJ)", showgrid=False),
                yaxis2=dict(title="평균기온 (℃)", overlaying='y', side='right', showgrid=False),
                xaxis=dict(tickformat="%d일", dtick="D1"),
                legend=dict(orientation="h", y=1.1), height=500, template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # 🟢 수정 2: 3차 다항식 수식 표시
            st.info(f"💡 **공급량 예측 모델 수식 (3차 다항식):** `{formula_str}`")
             
            with st.expander("📋 상세 데이터 보기"):
                df_down = df_final.copy()
                df_down['일자'] = df_down['일자'].dt.strftime('%Y-%m-%d')
                total_supply, avg_temp = df_down['공급량(MJ)'].sum(), df_down['평균기온(℃)'].mean()
                row_subtotal = pd.DataFrame([{'일자': '소계', '공급량(MJ)': total_supply, '평균기온(℃)': avg_temp, '구분': '-', '데이터출처': '-'}])
                df_down = pd.concat([df_down, row_subtotal], ignore_index=True)
                df_down['공급량(MJ)'] = df_down['공급량(MJ)'].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else "")
                df_down['평균기온(℃)'] = df_down['평균기온(℃)'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "")
                st.dataframe(df_down, use_container_width=True)
                st.download_button("📥 예측 결과 다운로드 (CSV)", df_down.to_csv(index=False).encode('utf-8-sig'), f"MM_{target_year}_{target_month}_forecast.csv", "text/csv")
    else:
        st.success("✅ 해당 월의 모든 실적이 확정되었습니다.")
        df_view = df_merged.copy()
        df_view['일자'] = df_view['일자'].dt.strftime('%Y-%m-%d')
        total_supply, avg_temp = df_view['공급량(MJ)'].sum(), df_view['평균기온(℃)'].mean()
        row_subtotal = pd.DataFrame([{'일자': '소계', '공급량(MJ)': total_supply, '평균기온(℃)': avg_temp, '구분': '-'}])
        df_view = pd.concat([df_view, row_subtotal], ignore_index=True)
        df_view['공급량(MJ)'] = df_view['공급량(MJ)'].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else "")
        df_view['평균기온(℃)'] = df_view['평균기온(℃)'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "")
        st.dataframe(df_view, use_container_width=True)

if __name__ == "__main__":
    main()
