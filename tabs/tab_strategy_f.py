"""
Strategy F v6.0 - Momentum & Trend Continuation Features

核心改變:
從「單點反轉」改為「趨勢延續」特徵,目標讓做多模型有效

新特徵:
1. 連續陽線/陰線計數 (趨勢持續性)
2. 價格突破追蹤 (高低點突破)
3. EMA排列強度 (多空力道)
4. 成交量趨勢斜率 (資金流向)
5. 動量指標組合 (ROC, ADX)

保留:
- TP/SL 結果標籤 (已驗證有效)
- 獨立多空模型
- 無冷卻期/趨勢過濾
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score
from sklearn.utils.class_weight import compute_sample_weight
import plotly.graph_objects as go

from backtesting.tick_level_engine import TickLevelBacktestEngine
from data.binance_loader import BinanceDataLoader


class MomentumMLv6:
    def __init__(self, lookback=10):
        self.lookback = lookback
        self.scaler_long = StandardScaler()
        self.scaler_short = StandardScaler()
        self.model_long = None
        self.model_short = None
        self.feature_names = []

    def calculate_indicators(self, df):
        df = df.copy()
        # 基礎
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(abs(df['high'] - df['close'].shift(1)),
                       abs(df['low'] - df['close'].shift(1)))
        )
        df['atr'] = df['tr'].rolling(14).mean()
        
        # EMA
        df['ema8'] = df['close'].ewm(span=8).mean()
        df['ema20'] = df['close'].ewm(span=20).mean()
        df['ema50'] = df['close'].ewm(span=50).mean()
        
        # EMA 排列強度 (趨勢力道)
        df['ema_spacing_8_20'] = (df['ema8'] - df['ema20']) / (df['atr'] + 1e-8)
        df['ema_spacing_20_50'] = (df['ema20'] - df['ema50']) / (df['atr'] + 1e-8)
        df['ema_alignment_bull'] = ((df['ema8'] > df['ema20']) & (df['ema20'] > df['ema50'])).astype(int)
        df['ema_alignment_bear'] = ((df['ema8'] < df['ema20']) & (df['ema20'] < df['ema50'])).astype(int)
        
        # 價格動量
        df['roc_5'] = df['close'].pct_change(5) * 100
        df['roc_10'] = df['close'].pct_change(10) * 100
        df['roc_20'] = df['close'].pct_change(20) * 100
        
        # 高低點突破
        df['high_20'] = df['high'].rolling(20).max()
        df['low_20'] = df['low'].rolling(20).min()
        df['breakout_high'] = (df['close'] > df['high_20'].shift(1)).astype(int)
        df['breakout_low'] = (df['close'] < df['low_20'].shift(1)).astype(int)
        df['dist_to_high'] = (df['close'] - df['high_20']) / (df['atr'] + 1e-8)
        df['dist_to_low'] = (df['close'] - df['low_20']) / (df['atr'] + 1e-8)
        
        # ADX (趨勢強度)
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        tr_smooth = df['tr'].rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / (tr_smooth + 1e-8))
        minus_di = 100 * (minus_dm.rolling(14).mean() / (tr_smooth + 1e-8))
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
        df['adx'] = dx.rolling(14).mean()
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di
        
        # 成交量趨勢
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-8)
        df['volume_trend'] = df['volume'].rolling(10).apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) == 10 else 0, raw=True)
        
        # BB
        df['bb_mid'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + 2 * bb_std
        df['bb_lower'] = df['bb_mid'] - 2 * bb_std
        df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['bb_mid'] + 1e-8)
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-8)))
        
        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        return df

    def momentum_features_row(self, df, i):
        """
        提取單點動量特徵 (不含回顧序列)
        """
        row = df.iloc[i]
        o, h, l, c = row['open'], row['high'], row['low'], row['close']
        body = c - o
        
        return [
            1 if body > 0 else -1,  # 當前K棒方向
            abs(body) / (h - l + 1e-8),  # 實體比例
            (h - l) / (row['atr'] + 1e-8),  # 波動幅度
        ]

    def build_sequence_features(self, df):
        df = self.calculate_indicators(df)
        df = df.reset_index(drop=True)
        
        # 當前值特徵
        current_features = [
            'rsi', 'macd_hist', 'bb_pct', 'bb_width',
            'volume_ratio', 'volume_trend',
            'ema_spacing_8_20', 'ema_spacing_20_50',
            'ema_alignment_bull', 'ema_alignment_bear',
            'roc_5', 'roc_10', 'roc_20',
            'breakout_high', 'breakout_low',
            'dist_to_high', 'dist_to_low',
            'adx', 'plus_di', 'minus_di',
        ]
        
        # 趨勢變化特徵 (3根K棒前 vs 現在)
        trend_features = ['rsi', 'macd_hist', 'roc_10', 'adx', 'volume_ratio']
        
        feature_names = []
        
        # 回顧序列: 連續陽線/陰線計數
        for lag in range(self.lookback, 0, -1):
            feature_names.extend([f"lag{lag}_dir", f"lag{lag}_body_ratio", f"lag{lag}_range_atr"])
        
        # 當前指標
        for col in current_features:
            feature_names.append(f"cur_{col}")
        
        # 趨勢變化
        for col in trend_features:
            feature_names.append(f"delta3_{col}")
        
        # 統計特徵 (過去10根)
        feature_names.extend([
            'consec_bull', 'consec_bear',  # 最大連續陽/陰線
            'bull_ratio_10', 'bear_ratio_10',  # 10根內陽/陰線比例
            'high_breakout_count_10', 'low_breakout_count_10',  # 突破次數
        ])
        
        self.feature_names = feature_names
        all_features, valid_indices = [], []
        
        for i in range(self.lookback, len(df)):
            row_feats = []
            
            # 回顧序列
            for lag in range(self.lookback, 0, -1):
                idx = i - lag
                row_feats.extend(self.momentum_features_row(df, idx))
            
            # 當前指標
            for col in current_features:
                v = df.iloc[i][col]
                row_feats.append(float(v) if not np.isnan(float(v)) else 0)
            
            # 趨勢變化
            for col in trend_features:
                v = df.iloc[i][col] - df.iloc[i - 3][col]
                row_feats.append(float(v) if not np.isnan(float(v)) else 0)
            
            # 統計特徵
            last_10 = df.iloc[i-9:i+1]
            directions = [(last_10.iloc[j]['close'] > last_10.iloc[j]['open']) for j in range(len(last_10))]
            
            # 最大連續陽/陰線
            consec_bull = consec_bear = 0
            max_bull = max_bear = 0
            for d in directions:
                if d:
                    consec_bull += 1; consec_bear = 0; max_bull = max(max_bull, consec_bull)
                else:
                    consec_bear += 1; consec_bull = 0; max_bear = max(max_bear, consec_bear)
            
            bull_ratio = sum(directions) / len(directions)
            bear_ratio = 1 - bull_ratio
            
            high_breakout_count = int(last_10['breakout_high'].sum())
            low_breakout_count = int(last_10['breakout_low'].sum())
            
            row_feats.extend([
                float(max_bull), float(max_bear),
                float(bull_ratio), float(bear_ratio),
                float(high_breakout_count), float(low_breakout_count),
            ])
            
            all_features.append(row_feats)
            valid_indices.append(i)
        
        X = pd.DataFrame(all_features, columns=feature_names)
        df_aligned = df.iloc[valid_indices].reset_index(drop=True)
        return X, df_aligned

    def build_trade_outcome_labels(self, df_aligned, sl_atr_mult=1.0, tp_atr_mult=3.0, max_lookahead=30):
        y_long, y_short, is_valid = [], [], []
        highs = df_aligned['high'].values
        lows = df_aligned['low'].values
        closes = df_aligned['close'].values
        atrs = df_aligned['atr'].values
        for i in range(len(df_aligned)):
            atr = atrs[i]; close = closes[i]
            if np.isnan(atr) or atr <= 0 or close <= 0:
                y_long.append(0); y_short.append(0); is_valid.append(False); continue
            long_sl = close - sl_atr_mult * atr
            long_tp = close + tp_atr_mult * atr
            short_sl = close + sl_atr_mult * atr
            short_tp = close - tp_atr_mult * atr
            long_outcome = None; short_outcome = None
            for j in range(i + 1, min(i + 1 + max_lookahead, len(df_aligned))):
                h = highs[j]; l = lows[j]
                if long_outcome is None:
                    if l <= long_sl and h >= long_tp: long_outcome = 0
                    elif l <= long_sl: long_outcome = 0
                    elif h >= long_tp: long_outcome = 1
                if short_outcome is None:
                    if h >= short_sl and l <= short_tp: short_outcome = 0
                    elif h >= short_sl: short_outcome = 0
                    elif l <= short_tp: short_outcome = 1
                if long_outcome is not None and short_outcome is not None: break
            if long_outcome is None or short_outcome is None:
                y_long.append(0); y_short.append(0); is_valid.append(False)
            else:
                y_long.append(long_outcome); y_short.append(short_outcome); is_valid.append(True)
        return pd.Series(y_long), pd.Series(y_short), pd.Series(is_valid)

    def train(self, df, sl_atr_mult=1.0, tp_atr_mult=3.0, max_lookahead=30, model_type='GBM'):
        X, df_aligned = self.build_sequence_features(df)
        y_long, y_short, is_valid = self.build_trade_outcome_labels(
            df_aligned, sl_atr_mult, tp_atr_mult, max_lookahead
        )
        mask = is_valid.values
        X_v = X[mask].reset_index(drop=True)
        yl_v = y_long[mask].reset_index(drop=True)
        ys_v = y_short[mask].reset_index(drop=True)
        split = int(len(X_v) * 0.8)
        X_tr, X_te = X_v.iloc[:split], X_v.iloc[split:]
        yl_tr, yl_te = yl_v.iloc[:split], yl_v.iloc[split:]
        ys_tr, ys_te = ys_v.iloc[:split], ys_v.iloc[split:]

        def make_model(mtype):
            if mtype == 'GBM':
                return GradientBoostingClassifier(
                    n_estimators=300, max_depth=4, learning_rate=0.04,
                    subsample=0.8, min_samples_leaf=20, random_state=42
                )
            return RandomForestClassifier(
                n_estimators=300, max_depth=5, min_samples_leaf=20,
                class_weight='balanced', random_state=42, n_jobs=-1
            )

        self.scaler_long.fit(X_tr)
        Xl_tr = self.scaler_long.transform(X_tr)
        Xl_te = self.scaler_long.transform(X_te)
        self.model_long = make_model(model_type)
        self.model_long.fit(Xl_tr, yl_tr, sample_weight=compute_sample_weight('balanced', yl_tr))

        self.scaler_short.fit(X_tr)
        Xs_tr = self.scaler_short.transform(X_tr)
        Xs_te = self.scaler_short.transform(X_te)
        self.model_short = make_model(model_type)
        self.model_short.fit(Xs_tr, ys_tr, sample_weight=compute_sample_weight('balanced', ys_tr))

        yl_pred = self.model_long.predict(Xl_te)
        ys_pred = self.model_short.predict(Xs_te)
        return {
            'train_size': len(X_tr), 'test_size': len(X_te),
            'long_prec': precision_score(yl_te, yl_pred, zero_division=0),
            'long_rec': recall_score(yl_te, yl_pred, zero_division=0),
            'short_prec': precision_score(ys_te, ys_pred, zero_division=0),
            'short_rec': recall_score(ys_te, ys_pred, zero_division=0),
            'long_base_rate': float(yl_te.mean() * 100),
            'short_base_rate': float(ys_te.mean() * 100),
            'valid_samples': int(mask.sum()),
            'sl_atr': sl_atr_mult, 'tp_atr': tp_atr_mult,
        }

    def generate_signals(
        self, df, leverage: float,
        direction='Both',
        conf_long=0.50, conf_short=0.50,
        prob_margin=0.00,
        sizing_mode='Risk',
        fixed_position_pct=0.3,
        risk_per_trade_pct=0.5,
        max_position_pct=0.5,
        stop_atr_mult=1.0,
        tp_atr_mult=3.0,
    ):
        X, df_aligned = self.build_sequence_features(df)
        Xl = self.scaler_long.transform(X)
        Xs = self.scaler_short.transform(X)
        proba_long = self.model_long.predict_proba(Xl)[:, 1]
        proba_short = self.model_short.predict_proba(Xs)[:, 1]
        signals = []
        
        for i in range(len(df_aligned)):
            atr = float(df_aligned.iloc[i]['atr'])
            close = float(df_aligned.iloc[i]['close'])
            pl = float(proba_long[i])
            ps = float(proba_short[i])
            sig = 0; sl = np.nan; tp = np.nan; reason = ""; pos_pct = 1.0
            
            if atr > 0 and close > 0:
                long_ok = (direction in ('Both', 'Long')) and (pl >= conf_long) and ((pl - ps) >= prob_margin)
                short_ok = (direction in ('Both', 'Short')) and (ps >= conf_short) and ((ps - pl) >= prob_margin)
                
                if long_ok and pl >= ps:
                    sig = 1
                    sl = close - atr * stop_atr_mult
                    tp = close + atr * tp_atr_mult
                    reason = f"L(p={pl:.2f})"
                elif short_ok and ps > pl:
                    sig = -1
                    sl = close + atr * stop_atr_mult
                    tp = close - atr * tp_atr_mult
                    reason = f"S(p={ps:.2f})"
                
                if sig != 0:
                    if sizing_mode == 'Fixed':
                        pos_pct = float(fixed_position_pct)
                    else:
                        atr_pct = atr / close
                        denom = max(atr_pct * stop_atr_mult * leverage, 1e-6)
                        pos_pct = float(np.clip((risk_per_trade_pct / 100.0) / denom, 0.01, max_position_pct))
            
            signals.append({
                'signal': sig, 'reason': reason,
                'stop_loss': sl, 'take_profit': tp,
                'position_size': pos_pct if sig != 0 else 1.0,
            })
        
        empty = [{'signal': 0, 'reason': '', 'stop_loss': np.nan,
                  'take_profit': np.nan, 'position_size': 1.0}] * self.lookback
        return pd.DataFrame(empty + signals), df_aligned

    def get_feature_importance(self, model='long', top_n=15):
        m = self.model_long if model == 'long' else self.model_short
        if m is None: return None
        return pd.Series(m.feature_importances_, index=self.feature_names).sort_values(ascending=False).head(top_n)


def _expected_pf(wr: float, tp: float, sl: float) -> float:
    if (1 - wr) * sl < 1e-9: return float('inf')
    return (wr * tp) / ((1 - wr) * sl)


def render_strategy_f_tab(loader, symbol_selector):
    st.header("策略 F: 動量趨勢 AI v6.0")

    with st.expander("ℹ️ v6.0 核心改進", expanded=True):
        st.markdown("""
        **問題診斷**: v5 影線特徵適合做空反轉,但做多模型失敗
        
        **v6 解決方案**:
        - ✖️ 移除影線特徵 (单點反轉)
        - ✅ 新增連續陽/陰線計數 (趨勢持續性)
        - ✅ 新增高低點突破追蹤 (突破交易)
        - ✅ 新增EMA排列強度 (多空力道)
        - ✅ 新增成交量趨勢斜率 (資金流向)
        - ✅ 新增 ADX + ROC (動量組合)
        
        **目標**: 讓做多模型 Precision > 基準勝率,做空維持優勢
        """)

    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**數據**")
        symbol_list = symbol_selector("strategy_f", multi=False)
        symbol = symbol_list[0]
        test_days = st.slider("測試天數", 14, 90, 30, key="test_f")
        train_days = st.slider("訓練天數", 60, 365, 180, key="train_f")
        timeframe = st.selectbox("時間周期", ['15m', '1h', '4h'], index=0, key="tf_f")
        bars_per_day = {'15m': 96, '1h': 24, '4h': 6}[timeframe]

    with col2:
        st.markdown("**交易**")
        capital = st.number_input("資金", 1000.0, 100000.0, 10000.0, 1000.0, key="cap_f")
        leverage = st.slider("槓桿", 1, 10, 3, key="lev_f")
        direction = st.radio("方向", ["Both", "Long", "Short"], index=0, key="dir_f")
        sizing_mode = st.radio("倉位", ["Risk", "Fixed"], index=0, key="sz_mode_f")
        if sizing_mode == "Fixed":
            fixed_position_pct = st.slider("固定倉位%", 5, 80, 30, 5, key="pos_fixed_f") / 100.0
            risk_per_trade_pct = 0.5
        else:
            fixed_position_pct = 0.3
            risk_per_trade_pct = st.slider("每筆風險%", 0.1, 2.0, 0.5, 0.1, key="risk_pct_f")
        max_position_pct = st.slider("最大倉位%", 10, 90, 50, 5, key="max_pos_f") / 100.0

    with col3:
        st.markdown("**AI**")
        model_type = st.radio("模型", ["GBM", "RandomForest"], key="model_f")
        lookback = st.slider("回顧K棒", 5, 20, 10, key="lb_f")
        sl_atr = st.slider("止損ATR", 0.5, 3.0, 1.0, 0.5, key="sl_f")
        tp_atr = st.slider("止盈ATR", 2.0, 8.0, 3.0, 0.5, key="tp_f")
        max_lookahead = st.slider("最大尋找K棒", 10, 50, 30, 5, key="ml_f")
        st.markdown("**過濾**")
        conf_long = st.slider("做多信心度", 0.50, 0.90, 0.50, 0.01, key="cl_f")
        conf_short = st.slider("做空信心度", 0.50, 0.90, 0.50, 0.01, key="cs_f")
        prob_margin = st.slider("機率差門檻", 0.00, 0.30, 0.00, 0.01, key="pm_f")

    breakeven_wr = sl_atr / (sl_atr + tp_atr)
    st.info(
        f"**TP={tp_atr}x / SL={sl_atr}x** | 盈虧平衡勝率: **{breakeven_wr*100:.1f}%** | "
        f"估計手續費損耗: ~{0.0012*leverage*100:.2f}%/筆 (槓桿{leverage}x)"
    )
    st.markdown("---")

    if st.button("訓練 + 回測", type="primary", use_container_width=True):
        prog = st.progress(0)
        stat = st.empty()
        try:
            stat.text("載入數據...")
            prog.progress(5)
            total_days = train_days + test_days + 5
            if isinstance(loader, BinanceDataLoader):
                end = datetime.now()
                df_all = loader.load_historical_data(symbol, timeframe, end - timedelta(days=total_days), end)
            else:
                df_all = loader.load_klines(symbol, timeframe)
                df_all = df_all.tail(total_days * bars_per_day)
            df_all = df_all.reset_index(drop=True)
            split_idx = int(len(df_all) * (train_days / total_days))
            df_train = df_all.iloc[:split_idx].reset_index(drop=True)
            df_test = df_all.iloc[split_idx:].reset_index(drop=True)
            st.info(f"訓練: {len(df_train)}根 | 測試: {len(df_test)}根")
            prog.progress(15)

            stat.text("訓練 AI (動量特徵)...")
            strategy = MomentumMLv6(lookback=lookback)
            result = strategy.train(
                df_train, sl_atr_mult=sl_atr, tp_atr_mult=tp_atr,
                max_lookahead=max_lookahead, model_type=model_type
            )
            prog.progress(55)

            st.markdown("### 訓練結果")
            c1, c2 = st.columns(2)
            check_long = '✅' if result['long_prec'] > result['long_base_rate']/100 else '❌'
            check_short = '✅' if result['short_prec'] > result['short_base_rate']/100 else '❌'
            
            with c1:
                st.markdown(f"**做多** (基準: {result['long_base_rate']:.1f}%)")
                lc1, lc2 = st.columns(2)
                lc1.metric("Precision", f"{result['long_prec']:.2%}")
                lc2.metric("Recall", f"{result['long_rec']:.2%}")
                epf_long = _expected_pf(result['long_prec'], tp_atr, sl_atr)
                st.caption(f"理論PF={epf_long:.2f} | {check_long}")
            with c2:
                st.markdown(f"**做空** (基準: {result['short_base_rate']:.1f}%)")
                sc1, sc2 = st.columns(2)
                sc1.metric("Precision", f"{result['short_prec']:.2%}")
                sc2.metric("Recall", f"{result['short_rec']:.2%}")
                epf_short = _expected_pf(result['short_prec'], tp_atr, sl_atr)
                st.caption(f"理論PF={epf_short:.2f} | {check_short}")

            with st.expander("特徵重要度"):
                tc1, tc2 = st.columns(2)
                for col_ui, mname, title in [(tc1, 'long', '做多'), (tc2, 'short', '做空')]:
                    imp = strategy.get_feature_importance(mname)
                    if imp is not None:
                        fig = go.Figure(go.Bar(x=imp.values[::-1], y=imp.index[::-1], orientation='h'))
                        fig.update_layout(height=400, title=title, margin=dict(l=180))
                        col_ui.plotly_chart(fig, use_container_width=True)

            prog.progress(65)
            stat.text("產生信號...")
            df_signals, df_aligned = strategy.generate_signals(
                df_test, leverage=leverage, direction=direction,
                conf_long=conf_long, conf_short=conf_short, prob_margin=prob_margin,
                sizing_mode=sizing_mode, fixed_position_pct=fixed_position_pct,
                risk_per_trade_pct=risk_per_trade_pct, max_position_pct=max_position_pct,
                stop_atr_mult=sl_atr, tp_atr_mult=tp_atr,
            )
            lc = (df_signals['signal'] == 1).sum()
            sc = (df_signals['signal'] == -1).sum()
            st.info(f"信號: {lc+sc} (L:{lc} S:{sc})")
            if lc + sc == 0:
                st.warning("無信號 - 降低信心度")
                prog.progress(100); return

            prog.progress(75)
            stat.text("回測...")
            df_for_bt = pd.concat([df_test.iloc[:lookback].reset_index(drop=True), df_aligned]).reset_index(drop=True)
            engine = TickLevelBacktestEngine(capital, leverage, 0.0006, 0.01, 100)
            metrics = engine.run_backtest(df_for_bt, df_signals)
            prog.progress(100)
            stat.text("完成")

            st.markdown("---")
            st.subheader("回測結果")
            c1, c2, c3, c4 = st.columns(4)
            pnl = metrics['final_equity'] - capital
            c1.metric("權益", f"${metrics['final_equity']:,.0f}", f"{pnl:+,.0f}")
            c1.metric("交易", metrics['total_trades'])
            ret = metrics['total_return_pct']
            c2.metric("總報酬", f"{ret:.1f}%")
            c2.metric("月化", f"{ret*30/test_days:.1f}%")
            wr = metrics['win_rate']
            pf = metrics['profit_factor']
            c3.metric("勝率", f"{wr:.1f}%", delta=f"vs 平衡 {breakeven_wr*100:.1f}%")
            c3.metric("盈虧比", f"{pf:.2f}")
            c4.metric("回撤", f"{metrics['max_drawdown_pct']:.1f}%")
            c4.metric("夏普", f"{metrics['sharpe_ratio']:.2f}")

            st.subheader("權益曲線")
            st.plotly_chart(engine.plot_equity_curve(), use_container_width=True)

            trades = engine.get_trades_dataframe()
            if not trades.empty:
                st.subheader("分方向分析")
                for direction_label, dir_key in [('做多', 'Long'), ('做空', 'Short')]:
                    dir_trades = trades[trades['direction'] == dir_key]
                    if len(dir_trades) == 0: continue
                    dir_wins = dir_trades[dir_trades['pnl_usdt'] > 0]
                    dir_wr = len(dir_wins) / len(dir_trades) * 100
                    dir_pnl = dir_trades['pnl_usdt'].sum()
                    dir_tp = (dir_trades['exit_reason'] == 'TP').sum()
                    dir_sl = (dir_trades['exit_reason'] == 'SL').sum()
                    status = "✅" if dir_pnl > 0 else "❌"
                    
                    if dir_key == 'Long':
                        model_prec = result['long_prec']
                    else:
                        model_prec = result['short_prec']
                    diff = dir_wr - model_prec * 100
                    diff_str = f"(模型預期 {model_prec*100:.1f}%, 差 {diff:+.1f}%)"
                    
                    st.markdown(
                        f"**{status} {direction_label}**: {len(dir_trades)}筆 | "
                        f"勝率 **{dir_wr:.1f}%** {diff_str} | "
                        f"損益 ${dir_pnl:+.0f} | TP:{dir_tp} SL:{dir_sl}"
                    )

                wins = trades[trades['pnl_usdt'] > 0]
                losses = trades[trades['pnl_usdt'] < 0]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("贏", len(wins))
                c2.metric("輸", len(losses))
                if len(wins): c3.metric("平均贏", f"${wins['pnl_usdt'].mean():.2f}")
                if len(losses): c4.metric("平均輸", f"${losses['pnl_usdt'].mean():.2f}")
                st.dataframe(
                    trades[['entry_time', 'direction', 'entry_price', 'exit_price', 'pnl_usdt', 'exit_reason']].tail(30),
                    use_container_width=True
                )
                csv = trades.to_csv(index=False).encode('utf-8')
                st.download_button("CSV", csv, f"{symbol}_momentum_v6_{datetime.now():%Y%m%d_%H%M}.csv", "text/csv")

        except Exception as e:
            st.error(f"錯: {e}")
            import traceback
            with st.expander("詳情"): st.code(traceback.format_exc())
