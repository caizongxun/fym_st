"""
策略E: Candle Pattern ML v4
重點改進 (針對回測仍大虧):
1) 風險定倉 (Risk-based position sizing): 用止損距離決定下單倉位,每筆固定風險%
2) 機率差過濾 (prob margin): 只有當多/空機率差夠大才進場
3) 維持 v3 的冷卻期 + EMA 趨勢過濾

為什麼: 你現在回測虧損的主要來源不是 "AI預測不準", 而是 "每筆曝險過大"。
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


class CandlePatternML:
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
            np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1)))
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
        df['ema_aligned_bull'] = ((df['ema8'] > df['ema20']) & (df['ema20'] > df['ema50'])).astype(int)
        df['ema_aligned_bear'] = ((df['ema8'] < df['ema20']) & (df['ema20'] < df['ema50'])).astype(int)

        low14 = df['low'].rolling(14).min()
        high14 = df['high'].rolling(14).max()
        df['stoch_k'] = 100 * (df['close'] - low14) / (high14 - low14 + 1e-8)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()

        tp = (df['high'] + df['low'] + df['close']) / 3
        tp_ma = tp.rolling(20).mean()
        tp_std = tp.rolling(20).std()
        df['cci'] = (tp - tp_ma) / (0.015 * tp_std + 1e-8)
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
            upper_shadow / total_range,
            lower_shadow / total_range,
            body_abs / total_range,
            1 if body > 0 else -1,
            1 if body > 0 else 0,
            1 if body_abs / total_range < 0.1 else 0,
            1 if (lower_shadow > 2 * body_abs and upper_shadow < body_abs) else 0,
            1 if (upper_shadow > 2 * body_abs and lower_shadow < body_abs) else 0,
            upper_shadow / (atr + 1e-8),
            lower_shadow / (atr + 1e-8),
            body_abs / (atr + 1e-8),
            total_range / (atr + 1e-8),
        ]

    def build_sequence_features(self, df):
        df = self.calculate_indicators(df)
        df = df.reset_index(drop=True)

        indicator_cols = [
            'rsi', 'macd_hist', 'bb_pct', 'bb_width',
            'volume_ratio', 'price_vs_ema20', 'ema_trend', 'ema_short_trend',
            'stoch_k', 'stoch_d', 'cci', 'williams_r',
            'ema_aligned_bull', 'ema_aligned_bear', 'vol_surge'
        ]

        candle_feat_names = ['up_shd', 'lo_shd', 'body', 'dir', 'bull', 'doji', 'hammer', 'star',
                             'up_atr', 'lo_atr', 'body_atr', 'range_atr']

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
                up_shadows.append(feats[0])
                lo_shadows.append(feats[1])
                bull_cnt += feats[4]
                bear_cnt += (1 - feats[4])
                doji_cnt += feats[5]

            for col in indicator_cols:
                v = df.iloc[i][col]
                row_feats.append(float(v) if not np.isnan(float(v)) else 0)

            for col in ['rsi', 'macd_hist', 'stoch_k']:
                v = df.iloc[i][col] - df.iloc[i - 3][col]
                row_feats.append(float(v) if not np.isnan(float(v)) else 0)

            row_feats.extend([
                float(np.sum(up_shadows)),
                float(np.sum(lo_shadows)),
                float(bull_cnt), float(bear_cnt), float(doji_cnt)
            ])

            all_features.append(row_feats)
            valid_indices.append(i)

        X = pd.DataFrame(all_features, columns=feature_names)
        df_aligned = df.iloc[valid_indices].reset_index(drop=True)
        return X, df_aligned

    def build_labels(self, df_aligned, lookahead=1, threshold_atr=0.4):
        y_long, y_short, y_raw = [], [], []
        for i in range(len(df_aligned)):
            if i + lookahead >= len(df_aligned):
                y_long.append(0); y_short.append(0); y_raw.append(0)
                continue
            atr = df_aligned.iloc[i]['atr']
            close = df_aligned.iloc[i]['close']
            future = df_aligned.iloc[i + 1:i + 1 + lookahead]
            future_high = future['high'].max()
            future_low = future['low'].min()
            future_close = future['close'].iloc[-1]

            up_move = (future_high - close) / (atr + 1e-8)
            down_move = (close - future_low) / (atr + 1e-8)
            net = (future_close - close) / (atr + 1e-8)

            is_up = 1 if up_move > threshold_atr and net > 0 else 0
            is_down = 1 if down_move > threshold_atr and net < 0 else 0

            y_long.append(is_up)
            y_short.append(is_down)
            y_raw.append(1 if is_up else (-1 if is_down else 0))

        return pd.Series(y_long), pd.Series(y_short), pd.Series(y_raw)

    def train(self, df, lookahead=1, threshold_atr=0.4, model_type='GBM'):
        X, df_aligned = self.build_sequence_features(df)
        y_long, y_short, y_raw = self.build_labels(df_aligned, lookahead, threshold_atr)

        split = int(len(X) * 0.8)
        X_tr, X_te = X.iloc[:split], X.iloc[split:]
        yl_tr, yl_te = y_long.iloc[:split], y_long.iloc[split:]
        ys_tr, ys_te = y_short.iloc[:split], y_short.iloc[split:]

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
            'train_size': len(X_tr),
            'test_size': len(X_te),
            'long_prec': precision_score(yl_te, yl_pred, zero_division=0),
            'long_rec': recall_score(yl_te, yl_pred, zero_division=0),
            'short_prec': precision_score(ys_te, ys_pred, zero_division=0),
            'short_rec': recall_score(ys_te, ys_pred, zero_division=0),
            'up_pct': y_raw.eq(1).mean() * 100,
            'dn_pct': y_raw.eq(-1).mean() * 100,
            'fl_pct': y_raw.eq(0).mean() * 100,
        }

    def generate_signals(
        self,
        df,
        leverage: float,
        conf_long=0.62,
        conf_short=0.62,
        prob_margin=0.06,
        sizing_mode='Risk',
        fixed_position_pct=0.5,
        risk_per_trade_pct=0.5,
        max_position_pct=0.6,
        stop_atr_mult=1.5,
        tp_atr_mult=3.0,
        cooldown_bars=5,
        use_trend_filter=True,
    ):
        """生成交易信號 + 倉位控制"""

        X, df_aligned = self.build_sequence_features(df)
        Xl = self.scaler_long.transform(X)
        Xs = self.scaler_short.transform(X)

        proba_long = self.model_long.predict_proba(Xl)[:, 1]
        proba_short = self.model_short.predict_proba(Xs)[:, 1]

        signals = []
        last_signal_bar = -cooldown_bars

        for i in range(len(df_aligned)):
            atr = float(df_aligned.iloc[i]['atr'])
            close = float(df_aligned.iloc[i]['close'])
            pl = float(proba_long[i])
            ps = float(proba_short[i])

            bull_trend = df_aligned.iloc[i]['ema_aligned_bull'] == 1
            bear_trend = df_aligned.iloc[i]['ema_aligned_bear'] == 1

            sig = 0
            sl = np.nan
            tp = np.nan
            reason = ""
            pos_pct = 1.0

            in_cooldown = (i - last_signal_bar) < cooldown_bars

            if not in_cooldown and atr > 0 and close > 0:
                # 機率差: 避免兩邊都差不多的狀況
                long_ok = (pl >= conf_long) and ((pl - ps) >= prob_margin)
                short_ok = (ps >= conf_short) and ((ps - pl) >= prob_margin)

                if use_trend_filter:
                    long_ok = long_ok and bull_trend
                    short_ok = short_ok and bear_trend

                if long_ok:
                    sig = 1
                    sl = close - atr * stop_atr_mult
                    tp = close + atr * tp_atr_mult
                    reason = f"LONG(p={pl:.2f},d={pl-ps:.2f})"
                    last_signal_bar = i
                elif short_ok:
                    sig = -1
                    sl = close + atr * stop_atr_mult
                    tp = close - atr * tp_atr_mult
                    reason = f"SHORT(p={ps:.2f},d={ps-pl:.2f})"
                    last_signal_bar = i

                if sig != 0:
                    if sizing_mode == 'Fixed':
                        pos_pct = float(fixed_position_pct)
                    else:
                        # 風險定倉: 讓每筆最大虧損(打到SL) ≈ risk_per_trade_pct 的帳戶資金
                        # loss_fraction ≈ (ATR/Close) * stop_mult * leverage * pos_pct
                        atr_pct = atr / close
                        denom = max(atr_pct * stop_atr_mult * leverage, 1e-6)
                        pos_pct = (risk_per_trade_pct / 100.0) / denom
                        pos_pct = float(np.clip(pos_pct, 0.01, max_position_pct))

            signals.append({
                'signal': sig,
                'reason': reason,
                'stop_loss': sl,
                'take_profit': tp,
                'position_size': pos_pct if sig != 0 else 1.0,
                'p_long': pl,
                'p_short': ps,
            })

        empty = [{
            'signal': 0,
            'reason': '',
            'stop_loss': np.nan,
            'take_profit': np.nan,
            'position_size': 1.0,
            'p_long': 0.0,
            'p_short': 0.0,
        }] * self.lookback

        return pd.DataFrame(empty + signals), df_aligned

    def get_feature_importance(self, model='long', top_n=15):
        m = self.model_long if model == 'long' else self.model_short
        if m is None:
            return None
        return pd.Series(m.feature_importances_, index=self.feature_names).sort_values(ascending=False).head(top_n)


def render_strategy_e_tab(loader, symbol_selector):
    st.header("策略 E: K棒影線 AI v4 (風險定倉)")

    st.info(
        """
        v4 改進重點:
        - 風險定倉: 每筆交易最大虧損固定為帳戶的 X% (用止損距離換算倉位)
        - 機率差過濾: 只有當多空機率差夠大才交易,避免隨機單
        - 保留冷卻期 + EMA 趨勢排列過濾
        """
    )

    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**數據**")
        symbol_list = symbol_selector("strategy_e", multi=False)
        symbol = symbol_list[0]
        test_days = st.slider("測試天數", 14, 90, 30, key="test_e")
        train_days = st.slider("訓練天數", 60, 365, 180, key="train_e")

    with col2:
        st.markdown("**交易**")
        capital = st.number_input("資金", 1000.0, 100000.0, 10000.0, 1000.0, key="cap_e")
        leverage = st.slider("槓桿", 1, 10, 3, key="lev_e")

        sizing_mode = st.radio("倉位模式", ["Risk", "Fixed"], index=0, key="sz_mode")
        if sizing_mode == "Fixed":
            fixed_position_pct = st.slider("固定倉位%", 5, 80, 30, 5, key="pos_fixed") / 100.0
            risk_per_trade_pct = 0.5
        else:
            fixed_position_pct = 0.3
            risk_per_trade_pct = st.slider("每筆風險% (打到SL)", 0.1, 2.0, 0.5, 0.1, key="risk_pct")

        max_position_pct = st.slider("最大倉位% (上限)", 10, 90, 60, 5, key="max_pos") / 100.0

    with col3:
        st.markdown("**AI設定**")
        model_type = st.radio("模型", ["GBM", "RandomForest"], key="model_e")
        lookback = st.slider("回顧K棒", 5, 20, 10, key="lb_e")
        lookahead = st.slider("預測K棒", 1, 5, 1, key="la_e")
        threshold_atr = st.slider("訊號門檻(ATR)", 0.2, 1.5, 0.4, 0.1, key="thr_e")

        conf_long = st.slider("做多信心度", 0.45, 0.90, 0.65, 0.01, key="cl_e")
        conf_short = st.slider("做空信心度", 0.45, 0.90, 0.65, 0.01, key="cs_e")
        prob_margin = st.slider("機率差門檻", 0.00, 0.25, 0.06, 0.01, key="pm_e")

        cooldown = st.slider("冷卻期(K棒)", 1, 30, 8, key="cd_e")
        trend_filter = st.checkbox("開啟EMA趨勢過濾", value=True, key="tf_e")

        st.markdown("**風控(價格端)**")
        stop_atr = st.slider("止損ATR", 0.5, 3.0, 1.5, 0.5, key="sl_e")
        tp_atr = st.slider("止盈ATR", 1.0, 6.0, 2.5, 0.5, key="tp_e")

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
                df_all = loader.load_historical_data(symbol, '15m', end - timedelta(days=total_days), end)
            else:
                df_all = loader.load_klines(symbol, '15m')
                df_all = df_all.tail(total_days * 96)

            df_all = df_all.reset_index(drop=True)
            split_idx = int(len(df_all) * (train_days / total_days))
            df_train = df_all.iloc[:split_idx].reset_index(drop=True)
            df_test = df_all.iloc[split_idx:].reset_index(drop=True)

            st.info(f"訓練: {len(df_train)}根 | 測試: {len(df_test)}根")
            prog.progress(15)

            stat.text("訓練AI...")
            strategy = CandlePatternML(lookback=lookback)
            result = strategy.train(
                df_train,
                lookahead=lookahead,
                threshold_atr=threshold_atr,
                model_type=model_type,
            )
            prog.progress(55)

            st.markdown("### 訓練結果")
            c1, c2, c3 = st.columns(3)
            c1.metric("訓練/測試", f"{result['train_size']} / {result['test_size']}")
            c2.metric("上漲/橫盤/下跌", f"{result['up_pct']:.0f}%/{result['fl_pct']:.0f}%/{result['dn_pct']:.0f}%")

            st.markdown("**做多模型**")
            lc1, lc2 = st.columns(2)
            lc1.metric("Precision", f"{result['long_prec']:.2%}")
            lc2.metric("Recall", f"{result['long_rec']:.2%}")

            st.markdown("**做空模型**")
            sc1, sc2 = st.columns(2)
            sc1.metric("Precision", f"{result['short_prec']:.2%}")
            sc2.metric("Recall", f"{result['short_rec']:.2%}")

            with st.expander("特徵重要度 Top15"):
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
                df_test,
                leverage=leverage,
                conf_long=conf_long,
                conf_short=conf_short,
                prob_margin=prob_margin,
                sizing_mode=sizing_mode,
                fixed_position_pct=fixed_position_pct,
                risk_per_trade_pct=risk_per_trade_pct,
                max_position_pct=max_position_pct,
                stop_atr_mult=stop_atr,
                tp_atr_mult=tp_atr,
                cooldown_bars=cooldown,
                use_trend_filter=trend_filter,
            )

            lc = (df_signals['signal'] == 1).sum()
            sc = (df_signals['signal'] == -1).sum()
            total_sig = lc + sc
            st.info(f"信號: {total_sig} (L:{lc} S:{sc})")

            if total_sig == 0:
                st.warning("無信號 - 降低信心度或降低機率差門檻")
                prog.progress(100)
                return

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
            c2.metric("月化", f"{ret * 30 / test_days:.1f}%")
            c3.metric("勝率", f"{metrics['win_rate']:.1f}%")
            c3.metric("盈虧比", f"{metrics['profit_factor']:.2f}")
            c4.metric("回撤", f"{metrics['max_drawdown_pct']:.1f}%")
            c4.metric("夏普", f"{metrics['sharpe_ratio']:.2f}")

            st.subheader("權益曲線")
            st.plotly_chart(engine.plot_equity_curve(), use_container_width=True)

            trades = engine.get_trades_dataframe()
            if not trades.empty:
                st.subheader("交易明細")
                wins = trades[trades['pnl_usdt'] > 0]
                losses = trades[trades['pnl_usdt'] < 0]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("贏", len(wins))
                c2.metric("輸", len(losses))
                if len(wins):
                    c3.metric("平均贏", f"${wins['pnl_usdt'].mean():.2f}")
                if len(losses):
                    c4.metric("平均輸", f"${losses['pnl_usdt'].mean():.2f}")

                st.dataframe(trades[['entry_time', 'direction', 'entry_price', 'exit_price', 'pnl_usdt', 'exit_reason']].tail(30), use_container_width=True)

                csv = trades.to_csv(index=False).encode('utf-8')
                st.download_button("CSV", csv, f"{symbol}_ml_v4_{datetime.now():%Y%m%d_%H%M}.csv", "text/csv")

        except Exception as e:
            st.error(f"錯: {e}")
            import traceback
            with st.expander("詳情"):
                st.code(traceback.format_exc())
