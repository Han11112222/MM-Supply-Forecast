import streamlit as st
import pandas as pd
import numpy as np
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

# 🟢 추가: 구글 시트 설정 (공급량_일일예측_실적관리)
GSHEET_ID = "13HrIz6OytYDykXeXzXJ02I6XbaKin1YaKBoO2kBd6Bs"
GSHEET_GID = "0"
GSHEET_CSV_URL = f"https://docs.google.com/spreadsheets/d/{GSHEET_ID}/export?format=csv&gid={GSHEET_GID}"

# 시트/엑셀의 컬럼명이 달라도 코드 내부 표준 컬럼명으로 통일
COL_ALIAS = {
    '최저': '최저기온(℃)',
    '최고': '최고기온(℃)',
    '최저기온': '최저기온(℃)',
    '최고기온': '최고기온(℃)',
    '평균기온': '평균기온(℃)',
    '공급량': '공급량(MJ)',
}


def set_korean_font():
    try:
        import matplotlib as mpl
        mpl.rcParams['axes.unicode_minus'] = False
        mpl.rc('font', family='Malgun Gothic')
    except Exception:
        pass


set_korean_font()


# ─────────────────────────────────────────────────────────
# 🟢 2. 데이터 로드 및 모델링
# ─────────────────────────────────────────────────────────
def normalize_df(df):
    """컬럼명 표준화 + 타입 정리 + 연/월/일 생성"""
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    df = df.rename(columns=COL_ALIAS)

    if '일자' in df.columns:
        df['일자'] = pd.to_datetime(df['일자'], errors='coerce')
        df = df.dropna(subset=['일자'])
        df['연'] = df['일자'].dt.year
        df['월'] = df['일자'].dt.month
        df['일'] = df['일자'].dt.day

    # 숫자 컬럼 정리 (쉼표 제거 등)
    for c in ['공급량(MJ)', '공급량(M3)', '평균기온(℃)', '최저기온(℃)', '최고기온(℃)']:
        if c in df.columns:
            df[c] = pd.to_numeric(
                df[c].astype(str).str.replace(',', '').str.strip().replace({'': np.nan, 'nan': np.nan}),
                errors='coerce'
            )
    return df


@st.cache_data(ttl=300, show_spinner="구글 시트에서 실적 불러오는 중...")
def load_gsheet_data(url):
    """🟢 구글 시트에서 실적(공급량/평균기온/최저/최고) 로드. (5분 캐시)"""
    try:
        df = pd.read_csv(url)
        return normalize_df(df), None
    except Exception as e:
        return pd.DataFrame(), str(e)


def read_uploaded(uploaded_file):
    """업로드된 엑셀/CSV 읽기"""
    try:
        if uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
    except Exception:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding='cp949')
    return normalize_df(df)


def load_local_fallback():
    """구글 시트 접근 실패 시 로컬 파일로 대체"""
    if os.path.exists("일일공급량_raw.xlsx"):
        return normalize_df(pd.read_excel("일일공급량_raw.xlsx"))
    return pd.DataFrame()


def train_models(df):
    """
    Model 1: 최저/최고 기온 -> 평균 기온 (선형회귀)
    Model 2: 평균 기온 -> 공급량 (3차 다항회귀)
    """
    df_t = df.dropna(subset=['최저기온(℃)', '최고기온(℃)', '평균기온(℃)'])
    model_temp = LinearRegression()
    model_temp.fit(df_t[['최저기온(℃)', '최고기온(℃)']], df_t['평균기온(℃)'])

    df_s = df.dropna(subset=['평균기온(℃)', '공급량(MJ)'])
    df_s = df_s[df_s['공급량(MJ)'] > 0]

    model_supply = make_pipeline(PolynomialFeatures(degree=3, include_bias=False), LinearRegression())
    model_supply.fit(df_s[['평균기온(℃)']], df_s['공급량(MJ)'])

    reg = model_supply.named_steps['linearregression']
    c0 = reg.intercept_
    c1, c2, c3 = reg.coef_
    formula = f"y = ({c3:.4e})x³ + ({c2:.4e})x² + ({c1:.4e})x + {c0:,.0f}"

    return model_temp, model_supply, formula


def get_past_stats(df_raw, target_month, method="3년 평균"):
    """과거 데이터 통계 추출 (빈 날짜 채우기용)"""
    stats_dict = {}

    df_past = df_raw[df_raw['월'] == target_month].dropna(subset=['최저기온(℃)', '최고기온(℃)']).copy()
    if df_past.empty:
        return stats_dict

    max_year = df_past['연'].max()
    if method == "과거 3년 평균":
        target_years = [max_year - 1, max_year - 2, max_year - 3]
    else:  # 전년도 실적
        target_years = [max_year - 1]

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

        # 🟢 수정: 실적은 구글 시트에서 자동 로드
        st.subheader("1. 실적 데이터 (구글 시트)")
        if st.button("🔄 구글 시트 새로고침", use_container_width=True):
            load_gsheet_data.clear()

        df_raw, gs_err = load_gsheet_data(GSHEET_CSV_URL)

        if df_raw.empty or '공급량(MJ)' not in df_raw.columns:
            st.error("⚠️ 구글 시트를 불러오지 못했습니다. (공유 설정: '링크가 있는 모든 사용자 - 뷰어' 확인)")
            if gs_err:
                st.caption(f"오류: {gs_err[:200]}")
            df_raw = load_local_fallback()
            if df_raw.empty:
                st.error("로컬 백업 파일(일일공급량_raw.xlsx)도 없습니다.")
                return
            st.warning("로컬 백업 파일(일일공급량_raw.xlsx)로 대체했습니다.")
        else:
            last_actual = df_raw.dropna(subset=['공급량(MJ)'])['일자'].max()
            st.success(f"✅ 시트 로드 완료 (마지막 실적일: {last_actual:%Y-%m-%d})")

        # 🟢 수정: 업로드 파일은 '오늘~말일 기온 예보' 입력용
        st.markdown("---")
        st.subheader("2. 예보 엑셀 (오늘~말일)")
        up_fcst = st.file_uploader(
            "일자 / 최저 / 최고 컬럼이 있는 파일 (선택)",
            type=['xlsx', 'csv'],
            help="업로드하면 예측 대상 날짜의 최저/최고 기온이 자동으로 채워지고, 표에서 추가 수정도 가능합니다."
        )
        df_fcst = pd.DataFrame()
        if up_fcst is not None:
            df_fcst = read_uploaded(up_fcst)
            need = {'일자', '최저기온(℃)', '최고기온(℃)'}
            if not need.issubset(df_fcst.columns):
                st.error("업로드 파일에 '일자', '최저(기온)', '최고(기온)' 컬럼이 필요합니다.")
                df_fcst = pd.DataFrame()

        # 학습 데이터 연도 선택
        st.markdown("---")
        st.subheader("🎯 학습 데이터 범위 설정")
        available_years = sorted(df_raw.dropna(subset=['공급량(MJ)'])['연'].unique().tolist(), reverse=True)
        default_years = available_years[:min(3, len(available_years))]

        selected_years = st.multiselect(
            "학습에 사용할 연도 선택",
            options=available_years,
            default=default_years,
            help="특정 연도(예: 2021년)를 제외하고 예측 모델을 학습시킬 수 있습니다."
        )

        df_train = df_raw[df_raw['연'].isin(selected_years)]

        if df_train.empty:
            st.warning("⚠️ 선택된 연도가 없습니다. 학습할 연도를 선택해주세요.")
            return

        st.markdown("---")
        st.subheader("📅 마감 대상 월 설정")
        today = datetime.date.today()
        target_year = int(st.number_input("연도 (Year)", value=today.year, step=1))
        target_month = int(st.number_input("월 (Month)", value=today.month, min_value=1, max_value=12, step=1))

        st.markdown("---")
        st.subheader("⚙️ 추정 옵션")
        fill_method = st.radio("미입력 구간(먼 미래) 기온 대체 방식", ["과거 3년 평균", "전년도 실적"])

    # 2. 모델 학습
    model_temp, model_supply, formula_str = train_models(df_train)

    # 3. 데이터 프레임 생성
    _, last_day = calendar.monthrange(target_year, target_month)
    dates = [datetime.date(target_year, target_month, d) for d in range(1, last_day + 1)]
    df_curr = pd.DataFrame({'일자': pd.to_datetime(dates)})

    # 4. 실적 매핑 (🟢 공급량이 실제로 입력된 날만 '실적'으로 인정)
    mask_month = (df_raw['연'] == target_year) & (df_raw['월'] == target_month) & df_raw['공급량(MJ)'].notna()
    df_actual = (df_raw[mask_month][['일자', '공급량(MJ)', '평균기온(℃)', '최저기온(℃)', '최고기온(℃)']]
                 .drop_duplicates(subset='일자', keep='last'))

    df_merged = pd.merge(df_curr, df_actual, on='일자', how='left')
    df_merged['구분'] = np.where(df_merged['공급량(MJ)'].notnull(), '실적', '예측대상')

    # 5. 사용자 입력
    missing_idx = df_merged[df_merged['구분'] == '예측대상'].index

    if len(missing_idx) > 0:
        st.info(f"📌 **{target_month}월**의 남은 **{len(missing_idx)}일**에 대한 예측을 수행합니다. "
                f"(학습 연도: {', '.join(map(str, selected_years))})")

        df_input = df_merged.loc[missing_idx, ['일자', '최저기온(℃)', '최고기온(℃)']].copy()

        # 🟢 추가: 업로드한 예보 파일 값으로 선채움
        if not df_fcst.empty:
            f = (df_fcst[['일자', '최저기온(℃)', '최고기온(℃)']]
                 .drop_duplicates(subset='일자', keep='last').set_index('일자'))
            for idx in df_input.index:
                d = df_input.at[idx, '일자']
                if d in f.index:
                    df_input.at[idx, '최저기온(℃)'] = f.at[d, '최저기온(℃)']
                    df_input.at[idx, '최고기온(℃)'] = f.at[d, '최고기온(℃)']
            n_filled = df_input[['최저기온(℃)', '최고기온(℃)']].notna().all(axis=1).sum()
            st.caption(f"📎 업로드 파일에서 {n_filled}일치 예보를 반영했습니다.")

        st.markdown("### 1️⃣ 기상청 예보 입력 (최저/최고)")
        st.caption("👇 아래 표를 수정할 수 있습니다. 비워두면 과거 패턴으로 대체됩니다.")

        edited_df = st.data_editor(
            df_input,
            num_rows="fixed", hide_index=True,
            column_config={
                "일자": st.column_config.DateColumn("날짜", format="MM-DD", disabled=True),
                "최저기온(℃)": st.column_config.NumberColumn("최저기온"),
                "최고기온(℃)": st.column_config.NumberColumn("최고기온"),
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
            df_final['데이터출처'] = np.where(df_final['구분'] == '실적', '실적', None)
            for idx in edited_df.index:
                t_min, t_max = edited_df.loc[idx, '최저기온(℃)'], edited_df.loc[idx, '최고기온(℃)']
                df_final.loc[idx, '최저기온(℃)'], df_final.loc[idx, '최고기온(℃)'] = t_min, t_max
                df_final.loc[idx, '데이터출처'] = '예보(입력)' if pd.notnull(t_min) and pd.notnull(t_max) else '과거패턴'

            # B. 빈값 채우기
            stats_map = get_past_stats(df_raw, target_month, fill_method)
            for i, row in df_final.iterrows():
                if row['구분'] == '예측대상' and (pd.isnull(row['최저기온(℃)']) or pd.isnull(row['최고기온(℃)'])):
                    md = (row['일자'].month, row['일자'].day)
                    if md in stats_map:
                        df_final.at[i, '최저기온(℃)'], df_final.at[i, '최고기온(℃)'] = stats_map[md]
                        df_final.at[i, '데이터출처'] = '과거패턴'

            # C. 평균기온 추정 (실적일은 시트의 실제 평균기온 사용)
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
            df_final['데이터출처'] = df_final['데이터출처'].fillna('과거패턴')

            # 6. 결과 시각화
            st.divider()
            st.subheader(f"📈 {target_year}년 {target_month}월 공급량 예측 결과")

            total_sum = df_final['공급량(MJ)'].sum()
            closed_sum = df_final[df_final['데이터출처'] == '실적']['공급량(MJ)'].sum()
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
                    fig.add_trace(go.Bar(x=df_sub['일자'], y=df_sub['공급량(MJ)'], name=f"{source}",
                                         marker_color=color_map[source]))

            # 🟢 수정: 평균기온을 실적(실제)과 추정으로 분리 표기
            df_t_act = df_final[df_final['데이터출처'] == '실적']
            df_t_est = df_final[df_final['데이터출처'] != '실적']

            if not df_t_act.empty:
                fig.add_trace(go.Scatter(x=df_t_act['일자'], y=df_t_act['평균기온(℃)'], name='평균기온(실적)',
                                         mode='lines+markers', line=dict(color='red', width=2),
                                         yaxis='y2'))
            if not df_t_est.empty:
                # 선이 끊기지 않도록 마지막 실적 지점을 추정선의 시작점으로 포함
                df_t_est_line = pd.concat([df_t_act.tail(1), df_t_est]) if not df_t_act.empty else df_t_est
                fig.add_trace(go.Scatter(x=df_t_est_line['일자'], y=df_t_est_line['평균기온(℃)'], name='평균기온(추정)',
                                         mode='lines+markers', line=dict(color='red', width=2, dash='dot'),
                                         yaxis='y2'))

            fig.update_layout(
                title=dict(text=f"일별 공급량 및 기온 추이 ({target_month}월)", font=dict(size=20)),
                yaxis=dict(title="공급량 (MJ)", showgrid=False),
                yaxis2=dict(title="평균기온 (℃)", overlaying='y', side='right', showgrid=False),
                xaxis=dict(tickformat="%d일", dtick="D1"),
                legend=dict(orientation="h", y=1.1), height=500, template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)

            st.info(f"💡 **공급량 예측 모델 수식 (3차 다항식):** {formula_str}")

            with st.expander("📋 상세 데이터 보기"):
                df_down = df_final.copy()
                df_down['일자'] = df_down['일자'].dt.strftime('%Y-%m-%d')
                total_supply, avg_temp = df_down['공급량(MJ)'].sum(), df_down['평균기온(℃)'].mean()
                row_subtotal = pd.DataFrame([{'일자': '소계', '공급량(MJ)': total_supply, '평균기온(℃)': avg_temp,
                                              '구분': '-', '데이터출처': '-'}])
                df_down = pd.concat([df_down, row_subtotal], ignore_index=True)
                df_down['공급량(MJ)'] = df_down['공급량(MJ)'].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else "")
                df_down['평균기온(℃)'] = df_down['평균기온(℃)'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "")
                st.dataframe(df_down, use_container_width=True)
                st.download_button("📥 예측 결과 다운로드 (CSV)", df_down.to_csv(index=False).encode('utf-8-sig'),
                                   f"MM_{target_year}_{target_month}_forecast.csv", "text/csv")
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
