"""
策略E: Candle Pattern ML v2
分離做多/做空模型 + 平衡類別 + 移除橫盤
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, precision_score, recall_score
from sklearn.utils.class_weight import compute_sample_weight
import plotly.graph_objects as go

from backtesting.tick_level_engine import TickLevelBacktestEngine
from data.binance_loader import BinanceDataLoader


class CandlePatternML:
    """
    K棒影線專家系統 v2
    - 分離做多/做空模型 (各自學習)
    - 移除橫盤K棒訓練
    - 平衡類別權重
    """

    def __init__(self, lookback=10):
        self.lookback = lookback
        self.scaler_long = StandardScaler()
        self.scaler_short = StandardScaler()
        self.model_long = None   # 預測是否上漲
        self.model_short = None  # 預測是否下跌
        self.feature_names = []

    # ==================== 指標計算 ====================

    def calculate_indicators(self, df):
        df = df.copy()
        # ATR
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(abs(df['high'] - df['close'].shift(1)),
                       abs(df['low'] - df['close'].shift(1)))
        )
        df['atr'] = df['tr'].rolling(14).mean()

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

        # Bollinger Bands
        df['bb_mid'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + 2 * bb_std
        df['bb_lower'] = df['bb_mid'] - 2 * bb_std
        df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['bb_mid'] + 1e-8)

        # Volume
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-8)

        # EMA Trend
        df['ema20'] = df['close'].ewm(span=20).mean()
        df['ema50'] = df['close'].ewm(span=50).mean()
        df['price_vs_ema20'] = (df['close'] - df['ema20']) / (df['atr'] + 1e-8)
        df['ema_trend'] = (df['ema20'] - df['ema50']) / (df['atr'] + 1e-8)

        # Stochastic
        low14 = df['low'].rolling(14).min()
        high14 = df['high'].rolling(14).max()
        df['stoch_k'] = 100 * (df['close'] - low14) / (high14 - low14 + 1e-8)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()

        # CCI
        tp = (df['high'] + df['low'] + df['close']) / 3
        tp_ma = tp.rolling(20).mean()
        tp_std = tp.rolling(20).std()
        df['cci'] = (tp - tp_ma) / (0.015 * tp_std + 1e-8)

        # Williams %R
        df['williams_r'] = -100 * (high14 - df['close']) / (high14 - low14 + 1e-8)

        return df

    def candle_features_row(self, o, h, l, c, atr):
        """單根K棒特徵"""
        body = c - o
        body_abs = abs(body) + 1e-8
        total_range = h - l + 1e-8
        upper_shadow = h - max(o, c)
        lower_shadow = min(o, c) - l

        return [
            upper_shadow / total_range,          # 上影線比
            lower_shadow / total_range,          # 下影線比
            body_abs / total_range,              # 實體比
            1 if body > 0 else -1,               # 方向
            1 if body > 0 else 0,                # 是否看漲
            1 if body_abs / total_range < 0.1 else 0,  # 十字線
            1 if (lower_shadow > 2*body_abs and upper_shadow < body_abs) else 0,  # 鉄锤
            1 if (upper_shadow > 2*body_abs and lower_shadow < body_abs) else 0,  # 流星
            upper_shadow / (atr + 1e-8),         # 上影線/ATR
            lower_shadow / (atr + 1e-8),         # 下影線/ATR
            body_abs / (atr + 1e-8),             # 實體/ATR
            total_range / (atr + 1e-8),          # 筏/ATR
        ]

    def build_sequence_features(self, df):
        """構建序列特徵"""
        df = self.calculate_indicators(df)
        df = df.reset_index(drop=True)

        indicator_cols = [
            'rsi', 'macd_hist', 'bb_pct', 'bb_width',
            'volume_ratio', 'price_vs_ema20', 'ema_trend',
            'stoch_k', 'stoch_d', 'cci', 'williams_r'
        ]

        per_candle_size = 12
        n_lags = self.lookback
        n_ind = len(indicator_cols)
        n_ind_trend = 3

        feature_names = []
        for lag in range(n_lags, 0, -1):
            for fname in ['up_shd', 'lo_shd', 'body', 'dir', 'bull', 'doji', 'hammer', 'star',
                          'up_atr', 'lo_atr', 'body_atr', 'range_atr']:
                feature_names.append(f"lag{lag}_{fname}")
        for col in indicator_cols:
            feature_names.append(f"cur_{col}")
        for col in ['rsi', 'macd_hist', 'stoch_k']:
            feature_names.append(f"trend3_{col}")
        # 額外影線結構特徵
        for col in ['up_sum', 'lo_sum', 'bull_cnt', 'bear_cnt', 'doji_cnt']:
            feature_names.append(f"roll10_{col}")

        self.feature_names = feature_names

        all_features = []
        valid_indices = []

        atrs = df['atr'].values

        for i in range(n_lags, len(df)):
            row_feats = []

            # 前10根K棒影線特徵
            up_shadows, lo_shadows, bull_cnt, bear_cnt, doji_cnt = [], [], 0, 0, 0
            for lag in range(n_lags, 0, -1):
                idx = i - lag
                o = df.iloc[idx]['open']
                h = df.iloc[idx]['high']
                l = df.iloc[idx]['low']
                c = df.iloc[idx]['close']
                atr = atrs[idx] if not np.isnan(atrs[idx]) else 1.0
                feats = self.candle_features_row(o, h, l, c, atr)
                row_feats.extend(feats)
                up_shadows.append(feats[0])
                lo_shadows.append(feats[1])
                bull_cnt += feats[4]
                bear_cnt += (1 - feats[4])
                doji_cnt += feats[5]

            # 當前指標
            for col in indicator_cols:
                v = df.iloc[i][col]
                row_feats.append(v if not np.isnan(v) else 0)

            # 指標趨勢 (3期變化)
            for col in ['rsi', 'macd_hist', 'stoch_k']:
                v = df.iloc[i][col] - df.iloc[i-3][col]
                row_feats.append(v if not np.isnan(v) else 0)

            # 影線結構小結
            row_feats.append(np.sum(up_shadows))   # 上影線總和
            row_feats.append(np.sum(lo_shadows))   # 下影線總和
            row_feats.append(bull_cnt)             # 看漲數
            row_feats.append(bear_cnt)             # 看跌數
            row_feats.append(doji_cnt)             # 十字線數

            all_features.append(row_feats)
            valid_indices.append(i)

        X = pd.DataFrame(all_features, columns=feature_names)
        df_aligned = df.iloc[valid_indices].reset_index(drop=True)
        return X, df_aligned

    def build_labels(self, df_aligned, lookahead=1, threshold_atr=0.4):
        """生成二元標籤"""
        y_long = []   # 1=上漲, 0=未上漲
        y_short = []  # 1=下跌, 0=未下跌
        y_raw = []    # -1/0/+1

        for i in range(len(df_aligned)):
            if i + lookahead >= len(df_aligned):
                y_long.append(0)
                y_short.append(0)
                y_raw.append(0)
                continue

            atr = df_aligned.iloc[i]['atr']
            current_close = df_aligned.iloc[i]['close']
            future = df_aligned.iloc[i+1:i+1+lookahead]

            future_high = future['high'].max()
            future_low = future['low'].min()
            future_close = future['close'].iloc[-1]

            up_move = (future_high - current_close) / (atr + 1e-8)
            down_move = (current_close - future_low) / (atr + 1e-8)
            net = (future_close - current_close) / (atr + 1e-8)

            is_up = 1 if up_move > threshold_atr and net > 0 else 0
            is_down = 1 if down_move > threshold_atr and net < 0 else 0

            y_long.append(is_up)
            y_short.append(is_down)
            y_raw.append(1 if is_up else (-1 if is_down else 0))

        return pd.Series(y_long), pd.Series(y_short), pd.Series(y_raw)

    def train(self, df, lookahead=1, threshold_atr=0.4, model_type='GBM'):
        X, df_aligned = self.build_sequence_features(df)
        y_long, y_short, y_raw = self.build_labels(df_aligned, lookahead, threshold_atr)

        # 訓練/測試分割 (80/20)
        split = int(len(X) * 0.8)
        X_tr, X_te = X.iloc[:split], X.iloc[split:]
        yl_tr, yl_te = y_long.iloc[:split], y_long.iloc[split:]
        ys_tr, ys_te = y_short.iloc[:split], y_short.iloc[split:]

        def make_model(mtype):
            if mtype == 'GBM':
                return GradientBoostingClassifier(
                    n_estimators=200, max_depth=4,
                    learning_rate=0.05, subsample=0.8, random_state=42
                )
            else:
                return RandomForestClassifier(
                    n_estimators=200, max_depth=6,
                    class_weight='balanced', random_state=42, n_jobs=-1
                )

        # 訓練做多模型
        self.scaler_long.fit(X_tr)
        Xl_tr_s = self.scaler_long.transform(X_tr)
        Xl_te_s = self.scaler_long.transform(X_te)
        self.model_long = make_model(model_type)
        # GBM手動平衡
        sw_long = compute_sample_weight('balanced', yl_tr)
        self.model_long.fit(Xl_tr_s, yl_tr, sample_weight=sw_long)

        # 訓練做空模型
        self.scaler_short.fit(X_tr)
        Xs_tr_s = self.scaler_short.transform(X_tr)
        Xs_te_s = self.scaler_short.transform(X_te)
        self.model_short = make_model(model_type)
        sw_short = compute_sample_weight('balanced', ys_tr)
        self.model_short.fit(Xs_tr_s, ys_tr, sample_weight=sw_short)

        # 測試評估
        yl_pred = self.model_long.predict(Xl_te_s)
        ys_pred = self.model_short.predict(Xs_te_s)

        long_prec = precision_score(yl_te, yl_pred, zero_division=0)
        long_rec  = recall_score(yl_te, yl_pred, zero_division=0)
        short_prec = precision_score(ys_te, ys_pred, zero_division=0)
        short_rec  = recall_score(ys_te, ys_pred, zero_division=0)

        dist = y_raw.value_counts().sort_index()

        return {
            'label_dist': dist,
            'train_size': len(X_tr),
            'test_size': len(X_te),
            'long_prec': long_prec,
            'long_rec': long_rec,
            'short_prec': short_prec,
            'short_rec': short_rec,
            'up_pct': y_raw.eq(1).mean() * 100,
            'dn_pct': y_raw.eq(-1).mean() * 100,
            'fl_pct': y_raw.eq(0).mean() * 100,
        }

    def generate_signals(self, df, conf_long=0.55, conf_short=0.55,
                         position_pct=0.7, stop_atr_mult=1.5, tp_atr_mult=2.5):
        X, df_aligned = self.build_sequence_features(df)

        Xl_s = self.scaler_long.transform(X)
        Xs_s = self.scaler_short.transform(X)

        proba_long  = self.model_long.predict_proba(Xl_s)[:, 1]   # P(上漲)
        proba_short = self.model_short.predict_proba(Xs_s)[:, 1]  # P(下跌)

        signals = []
        for i in range(len(df_aligned)):
            atr   = df_aligned.iloc[i]['atr']
            close = df_aligned.iloc[i]['close']
            pl = proba_long[i]
            ps = proba_short[i]

            sig = 0
            sl = np.nan
            tp = np.nan
            reason = ""

            if pl >= conf_long and pl > ps:
                sig = 1
                sl = close - atr * stop_atr_mult
                tp = close + atr * tp_atr_mult
                reason = f"LONG({pl:.2f})"
            elif ps >= conf_short and ps > pl:
                sig = -1
                sl = close + atr * stop_atr_mult
                tp = close - atr * tp_atr_mult
                reason = f"SHORT({ps:.2f})"

            signals.append({
                'signal': sig, 'reason': reason,
                'stop_loss': sl, 'take_profit': tp,
                'position_size': position_pct if sig != 0 else 1.0
            })

        empty = [{'signal': 0, 'reason': '', 'stop_loss': np.nan,
                  'take_profit': np.nan, 'position_size': 1.0}] * self.lookback
        return pd.DataFrame(empty + signals), df_aligned

    def get_feature_importance(self, model='long', top_n=20):
        m = self.model_long if model == 'long' else self.model_short
        if m is None:
            return None
        return pd.Series(m.feature_importances_, index=self.feature_names
                         ).sort_values(ascending=False).head(top_n)


def render_strategy_e_tab(loader, symbol_selector):
    st.header("策略 E: K棒影線 AI 預測系統 v2")

    st.info("""
    **Candle Pattern ML v2 - 分離做多/做空模型**

    改進:
    - 分離做多模型 + 做空模型 (各自專負一方)
    - 移除FLAT橫盤尚: 直接預測上漲/下跌機率
    - 類別平衡權重: 解決上漲/下跌樣本不平衡
    - 增加影線小結特徵: 10根K棒影線結構總結
    - CCI + Williams %R 新增指標
    """)

    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**數據**")
        symbol_list = symbol_selector("strategy_e", multi=False)
        symbol = symbol_list[0]
        test_days  = st.slider("測試天數", 14, 60, 30, key="test_e")
        train_days = st.slider("訓練天數", 60, 365, 90, key="train_e")

    with col2:
        st.markdown("**交易**")
        capital     = st.number_input("資金", 1000.0, 100000.0, 10000.0, 1000.0, key="cap_e")
        leverage    = st.slider("槓桿", 1, 10, 5, key="lev_e")
        position_pct = st.slider("倉位%", 30, 100, 70, 10, key="pos_e")

    with col3:
        st.markdown("**AI設定**")
        model_type    = st.radio("模型", ["GBM", "RandomForest"], key="model_e")
        lookback      = st.slider("回顧K棒", 5, 20, 10, key="lb_e")
        lookahead     = st.slider("預測K棒", 1, 5, 1, key="la_e")
        threshold_atr = st.slider("訊號門溻(ATR倍)", 0.2, 1.5, 0.4, 0.1, key="thr_e")
        conf_long     = st.slider("做多信心度", 0.3, 0.9, 0.55, 0.05, key="cl_e")
        conf_short    = st.slider("做空信心度", 0.3, 0.9, 0.55, 0.05, key="cs_e")
        st.markdown("**風控**")
        stop_atr = st.slider("止損ATR", 0.5, 3.0, 1.5, 0.5, key="sl_e")
        tp_atr   = st.slider("止盈ATR", 1.0, 5.0, 2.5, 0.5, key="tp_e")

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
                start = end - timedelta(days=total_days)
                df_all = loader.load_historical_data(symbol, '15m', start, end)
            else:
                df_all = loader.load_klines(symbol, '15m')
                df_all = df_all.tail(total_days * 96)

            df_all = df_all.reset_index(drop=True)
            split_idx = int(len(df_all) * (train_days / total_days))
            df_train = df_all.iloc[:split_idx].reset_index(drop=True)
            df_test  = df_all.iloc[split_idx:].reset_index(drop=True)
            st.info(f"訓練: {len(df_train)}根 | 測試: {len(df_test)}根")
            prog.progress(15)

            stat.text("訓練做多模型 + 做空模型...")
            strategy = CandlePatternML(lookback=lookback)
            result = strategy.train(
                df_train, lookahead=lookahead,
                threshold_atr=threshold_atr, model_type=model_type
            )
            prog.progress(55)

            # 顯示訓練結果
            st.markdown("### 訓練結果")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("訓練樣本", result['train_size'])
            c2.metric("測試樣本", result['test_size'])
            c3.metric("上漲/橫盤/下跌",
                      f"{result['up_pct']:.0f}%/{result['fl_pct']:.0f}%/{result['dn_pct']:.0f}%")

            st.markdown("**做多模型 (Long Model)**")
            lc1, lc2 = st.columns(2)
            lc1.metric("測試Precision", f"{result['long_prec']:.2%}")
            lc2.metric("測試Recall",    f"{result['long_rec']:.2%}")

            st.markdown("**做空模型 (Short Model)**")
            sc1, sc2 = st.columns(2)
            sc1.metric("測試Precision", f"{result['short_prec']:.2%}")
            sc2.metric("測試Recall",    f"{result['short_rec']:.2%}")

            # 特徵重要度
            with st.expander("特徵重要度"):
                tc1, tc2 = st.columns(2)
                for col, model_name, title in [(tc1,'long','做多'),(tc2,'short','做空')]:
                    imp = strategy.get_feature_importance(model_name, 15)
                    if imp is not None:
                        fig_imp = go.Figure(go.Bar(
                            x=imp.values[::-1], y=imp.index[::-1], orientation='h'
                        ))
                        fig_imp.update_layout(
                            height=400, title=f"{title} Top15",
                            margin=dict(l=180)
                        )
                        col.plotly_chart(fig_imp, use_container_width=True)

            prog.progress(65)

            stat.text("預測測試集...")
            df_signals, df_aligned = strategy.generate_signals(
                df_test,
                conf_long=conf_long, conf_short=conf_short,
                position_pct=position_pct / 100,
                stop_atr_mult=stop_atr, tp_atr_mult=tp_atr
            )

            signal_count = (df_signals['signal'] != 0).sum()
            long_cnt  = (df_signals['signal'] == 1).sum()
            short_cnt = (df_signals['signal'] == -1).sum()
            st.info(f"信號: {signal_count} (L:{long_cnt} S:{short_cnt})")

            if signal_count == 0:
                st.warning("無信號 - 降低信心度或ATR門溻")
                prog.progress(100)
                return

            prog.progress(75)
            stat.text("回測...")

            df_for_bt = pd.concat([
                df_test.iloc[:lookback].reset_index(drop=True),
                df_aligned
            ]).reset_index(drop=True)

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
            c3.metric("勝率", f"{wr:.1f}%")
            c3.metric("盈虧比", f"{pf:.2f}")
            c4.metric("回撤", f"{metrics['max_drawdown_pct']:.1f}%")
            c4.metric("夏普", f"{metrics['sharpe_ratio']:.2f}")

            st.markdown("---")
            if wr >= 50 and pf >= 1.5:
                st.success("✅ 有效!")
                st.balloons()
            elif ret > 0:
                st.info("⚠️ 有獲利 - 繼續優化")
            else:
                st.warning("❌ 調整參數")

            st.subheader("權益曲線")
            st.plotly_chart(engine.plot_equity_curve(), use_container_width=True)

            trades = engine.get_trades_dataframe()
            if not trades.empty:
                st.subheader("交易明細")
                wins   = trades[trades['pnl_usdt'] > 0]
                losses = trades[trades['pnl_usdt'] < 0]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("贏", len(wins))
                c2.metric("輸", len(losses))
                if len(wins) > 0:
                    c3.metric("平均贏", f"${wins['pnl_usdt'].mean():.2f}")
                if len(losses) > 0:
                    c4.metric("平均輸", f"${losses['pnl_usdt'].mean():.2f}")
                st.dataframe(
                    trades[['entry_time','direction','entry_price','exit_price','pnl_usdt','exit_reason']].tail(30),
                    use_container_width=True
                )
                csv = trades.to_csv(index=False).encode('utf-8')
                st.download_button("CSV", csv,
                    f"{symbol}_candle_ml_{datetime.now():%Y%m%d_%H%M}.csv", "text/csv")

        except Exception as e:
            st.error(f"錯: {e}")
            import traceback
            with st.expander("詳情"):
                st.code(traceback.format_exc())
