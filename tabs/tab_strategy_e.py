"""
策略E: Candle Pattern ML
利用前10根K棒的上下影線+指標,預測當下和下一根K棒走勢
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from backtesting.tick_level_engine import TickLevelBacktestEngine
from data.binance_loader import BinanceDataLoader


class CandlePatternML:
    """
    K棒影線專家系統
    
    學習前10根K棒的模式,預測下一根方向
    """
    
    def __init__(self, lookback=10):
        self.lookback = lookback
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = []
    
    # ==================== 影線特徵 ====================
    
    def candle_features(self, row):
        """單根K棒特徵"""
        o, h, l, c = row['open'], row['high'], row['low'], row['close']
        
        body = c - o
        body_abs = abs(body)
        total_range = h - l if h != l else 1e-8
        
        upper_shadow = h - max(o, c)  # 上影線
        lower_shadow = min(o, c) - l  # 下影線
        
        return {
            # 影線比例
            'upper_shadow_ratio': upper_shadow / total_range,
            'lower_shadow_ratio': lower_shadow / total_range,
            'body_ratio': body_abs / total_range,
            
            # 方向
            'direction': 1 if body > 0 else -1,
            'is_bullish': 1 if body > 0 else 0,
            
            # 相對大小 (需要ATR標準化)
            'body_size': body_abs,
            'total_size': total_range,
            
            # 影線強度
            'upper_shadow_abs': upper_shadow,
            'lower_shadow_abs': lower_shadow,
            
            # K棒特性
            'is_doji': 1 if body_abs / total_range < 0.1 else 0,
            'is_hammer': 1 if (lower_shadow > 2 * body_abs and upper_shadow < body_abs) else 0,
            'is_shooting_star': 1 if (upper_shadow > 2 * body_abs and lower_shadow < body_abs) else 0,
        }
    
    def calculate_indicators(self, df):
        """計算技術指標"""
        df = df.copy()
        
        # ATR
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(14).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + gain / loss))
        
        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_mid'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
        df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
        
        # Volume
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # EMA趨勢
        df['ema20'] = df['close'].ewm(span=20).mean()
        df['ema50'] = df['close'].ewm(span=50).mean()
        df['price_vs_ema20'] = (df['close'] - df['ema20']) / df['atr']
        df['ema_trend'] = (df['ema20'] - df['ema50']) / df['atr']
        
        # Stochastic
        low_14 = df['low'].rolling(14).min()
        high_14 = df['high'].rolling(14).max()
        df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14 + 1e-8)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        
        return df
    
    def build_sequence_features(self, df):
        """
        對每一根K棒,提取前10根K棒的影線+指標特徵
        """
        df = self.calculate_indicators(df)
        
        # 每根K棒的單棒特徵
        candle_feats = []
        for i in range(len(df)):
            row = df.iloc[i]
            feats = self.candle_features(row)
            candle_feats.append(feats)
        candle_feats_df = pd.DataFrame(candle_feats)
        
        # ATR標準化影線大小
        atr = df['atr'].values
        candle_feats_df['body_normalized'] = candle_feats_df['body_size'] / (atr + 1e-8)
        candle_feats_df['range_normalized'] = candle_feats_df['total_size'] / (atr + 1e-8)
        candle_feats_df['upper_shadow_normalized'] = candle_feats_df['upper_shadow_abs'] / (atr + 1e-8)
        candle_feats_df['lower_shadow_normalized'] = candle_feats_df['lower_shadow_abs'] / (atr + 1e-8)
        
        all_features = []
        feature_names = []
        
        # 定義要用到的特徵欄位
        per_candle_cols = [
            'upper_shadow_ratio', 'lower_shadow_ratio', 'body_ratio',
            'direction', 'is_bullish', 'is_doji', 'is_hammer', 'is_shooting_star',
            'body_normalized', 'range_normalized',
            'upper_shadow_normalized', 'lower_shadow_normalized'
        ]
        
        indicator_cols = [
            'rsi', 'macd_hist', 'bb_pct', 'bb_width',
            'volume_ratio', 'price_vs_ema20', 'ema_trend',
            'stoch_k', 'stoch_d'
        ]
        
        for i in range(self.lookback, len(df)):
            row_feats = []
            
            # 前10根K棒的K棒特徵
            for lag in range(self.lookback, 0, -1):  # lag=10,9,...,1
                for col in per_candle_cols:
                    val = candle_feats_df.iloc[i - lag][col]
                    row_feats.append(val if not np.isnan(val) else 0)
                    if i == self.lookback:
                        feature_names.append(f"lag{lag}_{col}")
            
            # 當前指標値 (lag=0)
            for col in indicator_cols:
                val = df.iloc[i][col]
                row_feats.append(val if not np.isnan(val) else 0)
                if i == self.lookback:
                    feature_names.append(f"cur_{col}")
            
            # 指標趨勢
            for col in ['rsi', 'macd_hist', 'stoch_k']:
                trend = df.iloc[i][col] - df.iloc[i-3][col]
                row_feats.append(trend if not np.isnan(trend) else 0)
                if i == self.lookback:
                    feature_names.append(f"trend3_{col}")
            
            all_features.append(row_feats)
        
        if not self.feature_names:
            self.feature_names = feature_names
        
        X = pd.DataFrame(all_features, columns=self.feature_names)
        return X, df.iloc[self.lookback:].reset_index(drop=True)
    
    def build_labels(self, df_aligned, lookahead=1, threshold_atr=0.5):
        """
        生成標籤:
        -1 = 下跌
         0 = 橫盤
        +1 = 上漲
        
        以ATR倍數為門溻
        """
        labels = []
        for i in range(len(df_aligned)):
            if i + lookahead >= len(df_aligned):
                labels.append(0)
                continue
            
            atr = df_aligned.iloc[i]['atr']
            current_close = df_aligned.iloc[i]['close']
            
            # 看未來lookahead根K棒的最高/最低和收盤
            future = df_aligned.iloc[i+1:i+1+lookahead]
            future_close = future['close'].iloc[-1]
            future_high = future['high'].max()
            future_low = future['low'].min()
            
            up_move = (future_high - current_close) / (atr + 1e-8)
            down_move = (current_close - future_low) / (atr + 1e-8)
            net_move = (future_close - current_close) / (atr + 1e-8)
            
            if net_move > threshold_atr:
                labels.append(1)   # 上漲
            elif net_move < -threshold_atr:
                labels.append(-1)  # 下跌
            else:
                labels.append(0)   # 橫盤
        
        return pd.Series(labels)
    
    def train(self, df, lookahead=1, threshold_atr=0.5, model_type='GBM'):
        """訓練模型"""
        X, df_aligned = self.build_sequence_features(df)
        y = self.build_labels(df_aligned, lookahead, threshold_atr)
        
        # 統計
        dist = y.value_counts().sort_index()
        
        # 訓練測試分割
        split = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        
        # 標準化
        self.scaler.fit(X_train)
        X_train_s = self.scaler.transform(X_train)
        X_test_s = self.scaler.transform(X_test)
        
        # 訓練
        if model_type == 'GBM':
            self.model = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42
            )
        else:
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=6,
                random_state=42,
                n_jobs=-1
            )
        
        self.model.fit(X_train_s, y_train)
        
        # 測試評估
        test_preds = self.model.predict(X_test_s)
        report = classification_report(y_test, test_preds, output_dict=True, zero_division=0)
        
        return {
            'label_dist': dist,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'report': report,
            'X_all': X,
            'df_aligned': df_aligned
        }
    
    def generate_signals(self, df, confidence=0.5, position_pct=0.7,
                         stop_atr_mult=1.5, tp_atr_mult=2.5):
        """生成交易信號"""
        X, df_aligned = self.build_sequence_features(df)
        X_scaled = self.scaler.transform(X)
        
        proba = self.model.predict_proba(X_scaled)
        classes = list(self.model.classes_)
        
        signals = []
        for i in range(len(df_aligned)):
            p = {c: proba[i][j] for j, c in enumerate(classes)}
            atr = df_aligned.iloc[i]['atr']
            close = df_aligned.iloc[i]['close']
            
            p_up = p.get(1, 0)
            p_down = p.get(-1, 0)
            p_flat = p.get(0, 0)
            
            sig = 0
            sl = np.nan
            tp = np.nan
            
            if p_up > confidence and p_up > p_down and p_up > p_flat:
                sig = 1
                sl = close - atr * stop_atr_mult
                tp = close + atr * tp_atr_mult
            elif p_down > confidence and p_down > p_up and p_down > p_flat:
                sig = -1
                sl = close + atr * stop_atr_mult
                tp = close - atr * tp_atr_mult
            
            signals.append({
                'signal': sig,
                'reason': f"ML_UP({p_up:.2f})" if sig == 1 else (f"ML_DN({p_down:.2f})" if sig == -1 else ""),
                'stop_loss': sl,
                'take_profit': tp,
                'position_size': position_pct if sig != 0 else 1.0,
                'p_up': p_up,
                'p_down': p_down,
                'p_flat': p_flat
            })
        
        # 前面lookback根补零
        empty = [{
            'signal': 0, 'reason': '', 'stop_loss': np.nan,
            'take_profit': np.nan, 'position_size': 1.0,
            'p_up': 0, 'p_down': 0, 'p_flat': 1
        }] * self.lookback
        
        return pd.DataFrame(empty + signals), df_aligned
    
    def get_feature_importance(self, top_n=20):
        """特徵重要度"""
        if self.model is None:
            return None
        imp = pd.Series(
            self.model.feature_importances_,
            index=self.feature_names
        ).sort_values(ascending=False).head(top_n)
        return imp


def render_strategy_e_tab(loader, symbol_selector):
    st.header("策略 E: K棒影線 AI 預測系統")
    
    st.info("""
    **Candle Pattern ML System**
    
    學習前10根K棒的影線模式 + 指標,預測下一根K棒走勢
    
    特徵:
    - 上影線/下影線/實體比例 (ATR標準化)
    - K棒形態: 錘子線/鉄锤/流星線
    - RSI/MACD/BB/Stochastic
    - 成交量比/EMA趨勢
    
    模型: GBM 或 RandomForest
    預測: 三分類 (+1/0/-1)
    """)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**數據**")
        symbol_list = symbol_selector("strategy_e", multi=False)
        symbol = symbol_list[0]
        test_days = st.slider("回測天數", 14, 90, 30, key="test_e")
        train_days = st.slider("訓練天數", 30, 180, 90, key="train_e")
        st.caption("訓練期和測試期不重疊")
    
    with col2:
        st.markdown("**交易**")
        capital = st.number_input("資金", 1000.0, 100000.0, 10000.0, 1000.0, key="cap_e")
        leverage = st.slider("槓桿", 1, 10, 5, key="lev_e")
        position_pct = st.slider("倉位%", 30, 100, 70, 10, key="pos_e")
    
    with col3:
        st.markdown("**AI設定**")
        model_type = st.radio("模型", ["GBM", "RandomForest"], key="model_e")
        lookback = st.slider("回顧K棒數", 5, 20, 10, key="lb_e")
        lookahead = st.slider("預測K棒數", 1, 5, 1, key="la_e")
        threshold_atr = st.slider("訊號門溻(ATR)", 0.2, 1.5, 0.5, 0.1, key="thr_e")
        confidence = st.slider("AI信心度", 0.3, 0.8, 0.45, 0.05, key="conf_e")
        
        st.markdown("**風控**")
        stop_atr = st.slider("止損ATR", 0.5, 3.0, 1.5, 0.5, key="sl_e")
        tp_atr = st.slider("止盈ATR", 1.0, 5.0, 2.5, 0.5, key="tp_e")
    
    st.markdown("---")
    
    if st.button("訓練 + 回測", type="primary", use_container_width=True):
        prog = st.progress(0)
        stat = st.empty()
        
        try:
            stat.text("載入數據...")
            prog.progress(5)
            
            total_days = train_days + test_days + 10
            
            if isinstance(loader, BinanceDataLoader):
                end = datetime.now()
                start = end - timedelta(days=total_days)
                df_all = loader.load_historical_data(symbol, '15m', start, end)
            else:
                df_all = loader.load_klines(symbol, '15m')
                df_all = df_all.tail(total_days * 96)
            
            df_all = df_all.reset_index(drop=True)
            
            # 分割訓練/測試
            split_idx = int(len(df_all) * (train_days / total_days))
            df_train = df_all.iloc[:split_idx].reset_index(drop=True)
            df_test = df_all.iloc[split_idx:].reset_index(drop=True)
            
            st.info(f"訓練: {len(df_train)}根 | 測試: {len(df_test)}根")
            prog.progress(15)
            
            stat.text("計算特徵...")
            strategy = CandlePatternML(lookback=lookback)
            
            stat.text("訓練AI模型...")
            result = strategy.train(
                df_train,
                lookahead=lookahead,
                threshold_atr=threshold_atr,
                model_type=model_type
            )
            prog.progress(50)
            
            # 顯示訓練結果
            st.markdown("### 訓練結果")
            c1, c2, c3 = st.columns(3)
            
            dist = result['label_dist']
            total_labels = dist.sum()
            c1.metric("訓練樣本", result['train_size'])
            c2.metric("測試樣本", result['test_size'])
            
            up_pct = dist.get(1, 0) / total_labels * 100
            dn_pct = dist.get(-1, 0) / total_labels * 100
            fl_pct = dist.get(0, 0) / total_labels * 100
            c3.metric("上漲/橫盤/下跌", f"{up_pct:.0f}% / {fl_pct:.0f}% / {dn_pct:.0f}%")
            
            report = result['report']
            rpt_cols = st.columns(3)
            for i, cls in enumerate(['1', '0', '-1']):
                if cls in report:
                    label = {"1": "UP", "0": "FLAT", "-1": "DOWN"}[cls]
                    rpt_cols[i].metric(
                        f"{label} Precision",
                        f"{report[cls]['precision']:.2%}",
                        f"Recall {report[cls]['recall']:.2%}"
                    )
            
            prog.progress(60)
            
            # 特徵重要度
            with st.expander("特徵重要度 Top20"):
                imp = strategy.get_feature_importance(20)
                if imp is not None:
                    fig_imp = go.Figure(go.Bar(
                        x=imp.values[::-1],
                        y=imp.index[::-1],
                        orientation='h'
                    ))
                    fig_imp.update_layout(
                        height=500,
                        title="特徵重要度 Top20",
                        xaxis_title="Importance",
                        margin=dict(l=200)
                    )
                    st.plotly_chart(fig_imp, use_container_width=True)
            
            stat.text("預測測試集...")
            prog.progress(70)
            
            df_signals, df_aligned = strategy.generate_signals(
                df_test,
                confidence=confidence,
                position_pct=position_pct / 100,
                stop_atr_mult=stop_atr,
                tp_atr_mult=tp_atr
            )
            
            signal_count = (df_signals['signal'] != 0).sum()
            long_cnt = (df_signals['signal'] == 1).sum()
            short_cnt = (df_signals['signal'] == -1).sum()
            
            st.info(f"信號: {signal_count} (L:{long_cnt} S:{short_cnt})")
            
            if signal_count == 0:
                st.warning("無信號 - 降低AI信心度或革點")
                prog.progress(100)
                return
            
            stat.text("回測...")
            prog.progress(80)
            
            # 回測引擎需要完整df
            # df_aligned是從第十根開始的,補齊前10根
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
            monthly = ret * 30 / test_days
            c2.metric("總報酬", f"{ret:.1f}%")
            c2.metric("月化", f"{monthly:.1f}%")
            
            wr = metrics['win_rate']
            pf = metrics['profit_factor']
            c3.metric("勝率", f"{wr:.1f}%")
            c3.metric("盈虧比", f"{pf:.2f}")
            
            dd = metrics['max_drawdown_pct']
            c4.metric("回撤", f"{dd:.1f}%")
            c4.metric("夏普", f"{metrics['sharpe_ratio']:.2f}")
            
            st.markdown("---")
            
            if wr >= 50 and pf >= 1.5:
                st.success("✅ K棒AI策略有效!")
                st.balloons()
            elif ret > 0:
                st.info("⚠️ 有獲利 - 繼續優化")
            else:
                st.warning("❌ 調整參數")
            
            st.markdown("---")
            st.subheader("權益曲線")
            fig = engine.plot_equity_curve()
            st.plotly_chart(fig, use_container_width=True)
            
            trades = engine.get_trades_dataframe()
            if not trades.empty:
                st.markdown("---")
                st.subheader("交易明細")
                wins = trades[trades['pnl_usdt'] > 0]
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
                st.download_button("CSV", csv, f"{symbol}_candle_ml_{datetime.now():%Y%m%d_%H%M}.csv", "text/csv")
        
        except Exception as e:
            st.error(f"錯: {e}")
            import traceback
            with st.expander("詳情"):
                st.code(traceback.format_exc())
