"""
策略E v5.2 - 精簡版

診斷: 過濾太強導致實際勝率 << 模型預期

修正:
1. 移除冷卻期 (直接用高信心度過濾)
2. 移除 EMA 趨勢過濾 (已經被特徵學到)
3. 只保留: 信心度 + 機率差
4. 預設 TP:SL = 4:1 (目標單筆獲利 > 手續費)
5. 預設只用做空模型 (Short Only)
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


class CandlePatternMLv5:
    def __init__(self, lookback=10):
        self.lookback = lookback
        self.scaler_long = StandardScaler()
        self.scaler_short = StandardScaler()
        self.model_long = None
        self.model_short = None
        self.feature_names = []

    def calculate_indicators(self, df):
        df = df.copy()
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(abs(df['high'] - df['close'].shift(1)),
                       abs(df['low'] - df['close'].shift(1)))
        )
        df['atr'] = df['tr'].rolling(14).mean()
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-8)))
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        df['bb_mid'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + 2 * bb_std
        df['bb_lower'] = df['bb_mid'] - 2 * bb_std
        df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['bb_mid'] + 1e-8)
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-8)
        df['ema8'] = df['close'].ewm(span=8).mean()
        df['ema20'] = df['close'].ewm(span=20).mean()
        df['ema50'] = df['close'].ewm(span=50).mean()
        df['price_vs_ema20'] = (df['close'] - df['ema20']) / (df['atr'] + 1e-8)
        df['ema_trend'] = (df['ema20'] - df['ema50']) / (df['atr'] + 1e-8)
        df['ema_short_trend'] = (df['ema8'] - df['ema20']) / (df['atr'] + 1e-8)
        low14 = df['low'].rolling(14).min()
        high14 = df['high'].rolling(14).max()
        df['stoch_k'] = 100 * (df['close'] - low14) / (high14 - low14 + 1e-8)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        tp_val = (df['high'] + df['low'] + df['close']) / 3
        tp_ma = tp_val.rolling(20).mean()
        tp_std = tp_val.rolling(20).std()
        df['cci'] = (tp_val - tp_ma) / (0.015 * tp_std + 1e-8)
        df['williams_r'] = -100 * (high14 - df['close']) / (high14 - low14 + 1e-8)
        df['vol_surge'] = (df['volume_ratio'] > 1.5).astype(int)
        return df

    def candle_features_row(self, o, h, l, c, atr):
        body = c - o
        body_abs = abs(body) + 1e-8
        total_range = h - l + 1e-8
        upper_shadow = h - max(o, c)
        lower_shadow = min(o, c) - l
        return [
            upper_shadow / total_range, lower_shadow / total_range,
            body_abs / total_range, 1 if body > 0 else -1,
            1 if body > 0 else 0,
            1 if body_abs / total_range < 0.1 else 0,
            1 if (lower_shadow > 2 * body_abs and upper_shadow < body_abs) else 0,
            1 if (upper_shadow > 2 * body_abs and lower_shadow < body_abs) else 0,
            upper_shadow / (atr + 1e-8), lower_shadow / (atr + 1e-8),
            body_abs / (atr + 1e-8), total_range / (atr + 1e-8),
        ]

    def build_sequence_features(self, df):
        df = self.calculate_indicators(df)
        df = df.reset_index(drop=True)
        indicator_cols = [
            'rsi', 'macd_hist', 'bb_pct', 'bb_width',
            'volume_ratio', 'price_vs_ema20', 'ema_trend', 'ema_short_trend',
            'stoch_k', 'stoch_d', 'cci', 'williams_r', 'vol_surge'
        ]
        candle_feat_names = ['up_shd', 'lo_shd', 'body', 'dir', 'bull', 'doji',
                             'hammer', 'star', 'up_atr', 'lo_atr', 'body_atr', 'range_atr']
        feature_names = []
        for lag in range(self.lookback, 0, -1):
            for fn in candle_feat_names:
                feature_names.append(f"lag{lag}_{fn}")
        for col in indicator_cols:
            feature_names.append(f"cur_{col}")
        for col in ['rsi', 'macd_hist', 'stoch_k']:
            feature_names.append(f"trend3_{col}")
        for col in ['up_sum', 'lo_sum', 'bull_cnt', 'bear_cnt', 'doji_cnt']:
            feature_names.append(f"roll10_{col}")
        self.feature_names = feature_names
        all_features, valid_indices = [], []
        atrs = df['atr'].values
        for i in range(self.lookback, len(df)):
            row_feats = []
            up_shadows, lo_shadows, bull_cnt, bear_cnt, doji_cnt = [], [], 0, 0, 0
            for lag in range(self.lookback, 0, -1):
                idx = i - lag
                o, h, l, c = df.iloc[idx][['open', 'high', 'low', 'close']]
                atr = atrs[idx] if not np.isnan(atrs[idx]) else 1.0
                feats = self.candle_features_row(o, h, l, c, atr)
                row_feats.extend(feats)
                up_shadows.append(feats[0]); lo_shadows.append(feats[1])
                bull_cnt += feats[4]; bear_cnt += (1 - feats[4]); doji_cnt += feats[5]
            for col in indicator_cols:
                v = df.iloc[i][col]
                row_feats.append(float(v) if not np.isnan(float(v)) else 0)
            for col in ['rsi', 'macd_hist', 'stoch_k']:
                v = df.iloc[i][col] - df.iloc[i - 3][col]
                row_feats.append(float(v) if not np.isnan(float(v)) else 0)
            row_feats.extend([
                float(np.sum(up_shadows)), float(np.sum(lo_shadows)),
                float(bull_cnt), float(bear_cnt), float(doji_cnt)
            ])
            all_features.append(row_feats)
            valid_indices.append(i)
        X = pd.DataFrame(all_features, columns=feature_names)
        df_aligned = df.iloc[valid_indices].reset_index(drop=True)
        return X, df_aligned

    def build_trade_outcome_labels(self, df_aligned, sl_atr_mult=1.0, tp_atr_mult=4.0, max_lookahead=30):
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

    def train(self, df, sl_atr_mult=1.0, tp_atr_mult=4.0, max_lookahead=30, model_type='GBM'):
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
        direction='Short',
        conf_long=0.60, conf_short=0.60,
        prob_margin=0.08,
        sizing_mode='Risk',
        fixed_position_pct=0.3,
        risk_per_trade_pct=0.5,
        max_position_pct=0.5,
        stop_atr_mult=1.0,
        tp_atr_mult=4.0,
    ):
        """
        v5.2: 移除冷卻期 + EMA趨勢過濾, 只保留信心度 + 機率差
        """
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


def render_strategy_e_tab(loader, symbol_selector):
    st.header("策略 E: K棒影線 AI v5.2 (精簡版)")

    with st.expander("ℹ️ v5.2 核心改進", expanded=True):
        st.markdown("""
        **v5.1 診斷**: 過濾太強導致實際勝率 28.3% << 模型預期 35.4%
        
        **v5.2 修正**:
        - ✖️ 移除冷卻期 (直接用高信心度控制信號數)
        - ✖️ 移除 EMA 趨勢過濾 (EMA 已被特徵學到)
        - ✅ 只保留: 信心度 + 機率差
        - ✅ TP:SL 預設 4:1 (單筆獲利 > 手續費)
        - ✅ 預設方向: Short Only
        - ✅ 最大尋找增加到 30 根 K 棒 (適應 4:1)
        
        **目標**: 讓實際勝率 ≈ 模型 Precision
        """)

    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**數據**")
        symbol_list = symbol_selector("strategy_e", multi=False)
        symbol = symbol_list[0]
        test_days = st.slider("測試天數", 14, 90, 30, key="test_e")
        train_days = st.slider("訓練天數", 60, 365, 180, key="train_e")
        timeframe = st.selectbox("時間周期", ['15m', '1h', '4h'], index=0, key="tf_e")
        bars_per_day = {'15m': 96, '1h': 24, '4h': 6}[timeframe]

    with col2:
        st.markdown("**交易**")
        capital = st.number_input("資金", 1000.0, 100000.0, 10000.0, 1000.0, key="cap_e")
        leverage = st.slider("槓桿", 1, 10, 3, key="lev_e")
        direction = st.radio("方向", ["Short", "Long", "Both"], index=0, key="dir_e")
        sizing_mode = st.radio("倉位", ["Risk", "Fixed"], index=0, key="sz_mode")
        if sizing_mode == "Fixed":
            fixed_position_pct = st.slider("固定倉位%", 5, 80, 30, 5, key="pos_fixed") / 100.0
            risk_per_trade_pct = 0.5
        else:
            fixed_position_pct = 0.3
            risk_per_trade_pct = st.slider("每筆風險%", 0.1, 2.0, 0.5, 0.1, key="risk_pct")
        max_position_pct = st.slider("最大倉位%", 10, 90, 50, 5, key="max_pos") / 100.0

    with col3:
        st.markdown("**AI**")
        model_type = st.radio("模型", ["GBM", "RandomForest"], key="model_e")
        lookback = st.slider("回顧K棒", 5, 20, 10, key="lb_e")
        sl_atr = st.slider("止損ATR", 0.5, 3.0, 1.0, 0.5, key="sl_e")
        tp_atr = st.slider("止盈ATR", 2.0, 8.0, 4.0, 0.5, key="tp_e")
        max_lookahead = st.slider("最大尋找K棒", 10, 50, 30, 5, key="ml_e")
        st.markdown("**過濾**")
        conf_long = st.slider("做多信心度", 0.50, 0.90, 0.60, 0.01, key="cl_e")
        conf_short = st.slider("做空信心度", 0.50, 0.90, 0.60, 0.01, key="cs_e")
        prob_margin = st.slider("機率差門檻", 0.00, 0.30, 0.08, 0.01, key="pm_e")

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

            stat.text("訓練 AI...")
            strategy = CandlePatternMLv5(lookback=lookback)
            result = strategy.train(
                df_train, sl_atr_mult=sl_atr, tp_atr_mult=tp_atr,
                max_lookahead=max_lookahead, model_type=model_type
            )
            prog.progress(55)

            st.markdown("### 訓練結果")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**做多** (基準: {result['long_base_rate']:.1f}%)")
                lc1, lc2 = st.columns(2)
                lc1.metric("Precision", f"{result['long_prec']:.2%}")
                lc2.metric("Recall", f"{result['long_rec']:.2%}")
                epf_long = _expected_pf(result['long_prec'], tp_atr, sl_atr)
                st.caption(f"理論PF={epf_long:.2f} | {'\u2705' if result['long_prec'] > result['long_base_rate']/100 else '\u274c'}")
            with c2:
                st.markdown(f"**做空** (基準: {result['short_base_rate']:.1f}%)")
                sc1, sc2 = st.columns(2)
                sc1.metric("Precision", f"{result['short_prec']:.2%}")
                sc2.metric("Recall", f"{result['short_rec']:.2%}")
                epf_short = _expected_pf(result['short_prec'], tp_atr, sl_atr)
                st.caption(f"理論PF={epf_short:.2f} | {'\u2705' if result['short_prec'] > result['short_base_rate']/100 else '\u274c'}")

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
                    
                    # 模型預期 vs 實際
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
                st.download_button("CSV", csv, f"{symbol}_ml_v52_{datetime.now():%Y%m%d_%H%M}.csv", "text/csv")

        except Exception as e:
            st.error(f"錯: {e}")
            import traceback
            with st.expander("詳情"): st.code(traceback.format_exc())
