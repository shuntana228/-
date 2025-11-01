#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
財務分析ツール - 証券コードから企業の財務分析結果をグラフ表示
Financial Analysis Tool - Display graphical financial analysis for companies by securities code
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mplfinance as mpf
from datetime import datetime, timedelta
import sys
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


class FinancialAnalyzer:
    """財務分析クラス"""

    def __init__(self, ticker_code):
        """
        初期化

        Args:
            ticker_code (str): 証券コード（例: "7203" for Toyota）
        """
        self.ticker_code = ticker_code
        self.ticker_symbol = self._format_ticker(ticker_code)
        self.stock = None
        self.info = None
        self.historical_data = None
        self.financials = None

    def _format_ticker(self, code):
        """
        証券コードをティッカーシンボルに変換
        日本株の場合は.Tを付加、それ以外はそのまま
        """
        # 4桁の数字の場合は日本株と判断
        if code.isdigit() and len(code) == 4:
            return f"{code}.T"
        return code

    def fetch_data(self, period="2y"):
        """
        データを取得

        Args:
            period (str): データ取得期間（例: "1y", "2y", "5y", "max"）
        """
        print(f"📊 証券コード {self.ticker_code} ({self.ticker_symbol}) のデータを取得中...")

        try:
            self.stock = yf.Ticker(self.ticker_symbol)
            self.info = self.stock.info
            self.historical_data = self.stock.history(period=period)

            # 財務データを取得
            try:
                self.financials = self.stock.financials
                self.balance_sheet = self.stock.balance_sheet
                self.cashflow = self.stock.cashflow
            except Exception as e:
                print(f"⚠️  財務諸表データの取得に失敗: {e}")
                self.financials = None

            if self.historical_data.empty:
                raise ValueError(f"データが見つかりません: {self.ticker_symbol}")

            print(f"✅ データ取得完了: {len(self.historical_data)} 件のレコード")
            return True

        except Exception as e:
            print(f"❌ エラー: {e}")
            return False

    def calculate_technical_indicators(self):
        """テクニカル指標を計算"""
        if self.historical_data is None or self.historical_data.empty:
            return

        df = self.historical_data.copy()

        # 移動平均線
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA25'] = df['Close'].rolling(window=25).mean()
        df['MA75'] = df['Close'].rolling(window=75).mean()

        # ボリンジャーバンド
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)

        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # 出来高移動平均
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()

        self.historical_data = df

    def calculate_roic(self):
        """
        ROIC (Return on Invested Capital) を計算
        投下資本利益率 = NOPAT / 投下資本

        Returns:
            float: ROIC (%)、計算できない場合はNone
        """
        try:
            if self.financials is None or self.balance_sheet is None:
                return None

            # NOPATの計算: Operating Income × (1 - Tax Rate)
            # 最新の財務データを使用（列の最初の要素）
            if 'Operating Income' in self.financials.index:
                operating_income = self.financials.loc['Operating Income'].iloc[0]
            elif 'EBIT' in self.financials.index:
                operating_income = self.financials.loc['EBIT'].iloc[0]
            else:
                return None

            # 実効税率を計算
            if 'Tax Provision' in self.financials.index and 'Pretax Income' in self.financials.index:
                tax_provision = self.financials.loc['Tax Provision'].iloc[0]
                pretax_income = self.financials.loc['Pretax Income'].iloc[0]
                if pretax_income != 0:
                    tax_rate = abs(tax_provision / pretax_income)
                else:
                    tax_rate = 0.25  # デフォルト税率
            else:
                tax_rate = 0.25  # デフォルト税率25%

            nopat = operating_income * (1 - tax_rate)

            # 投下資本の計算: Total Equity + Total Debt - Cash
            if 'Total Equity Gross Minority Interest' in self.balance_sheet.index:
                total_equity = self.balance_sheet.loc['Total Equity Gross Minority Interest'].iloc[0]
            elif 'Stockholders Equity' in self.balance_sheet.index:
                total_equity = self.balance_sheet.loc['Stockholders Equity'].iloc[0]
            else:
                return None

            # 総負債を取得
            if 'Total Debt' in self.balance_sheet.index:
                total_debt = self.balance_sheet.loc['Total Debt'].iloc[0]
            elif 'Long Term Debt' in self.balance_sheet.index and 'Current Debt' in self.balance_sheet.index:
                total_debt = self.balance_sheet.loc['Long Term Debt'].iloc[0] + self.balance_sheet.loc['Current Debt'].iloc[0]
            else:
                total_debt = 0

            # 現金を取得
            if 'Cash And Cash Equivalents' in self.balance_sheet.index:
                cash = self.balance_sheet.loc['Cash And Cash Equivalents'].iloc[0]
            else:
                cash = 0

            invested_capital = total_equity + total_debt - cash

            if invested_capital <= 0:
                return None

            roic = (nopat / invested_capital) * 100

            return roic

        except Exception as e:
            print(f"⚠️  ROIC計算エラー: {e}")
            return None

    def calculate_wacc(self):
        """
        WACC (Weighted Average Cost of Capital) を計算
        加重平均資本コスト = (E/V × Re) + (D/V × Rd × (1-T))

        Returns:
            float: WACC (%)、計算できない場合はNone
        """
        try:
            if not self.info or self.balance_sheet is None:
                return None

            # 時価総額（株主資本の市場価値）
            market_cap = self.info.get('marketCap', 0)
            if market_cap == 0:
                return None

            # 総負債を取得
            if 'Total Debt' in self.balance_sheet.index:
                total_debt = self.balance_sheet.loc['Total Debt'].iloc[0]
            elif 'Long Term Debt' in self.balance_sheet.index and 'Current Debt' in self.balance_sheet.index:
                total_debt = self.balance_sheet.loc['Long Term Debt'].iloc[0] + self.balance_sheet.loc['Current Debt'].iloc[0]
            else:
                total_debt = 0

            # 企業価値
            enterprise_value = market_cap + total_debt

            if enterprise_value <= 0:
                return None

            # 株主資本コスト（Re）の推定
            # 簡易的にCAPMを使用: Re = Rf + β × (Rm - Rf)
            # ここでは、より単純に過去のリターンとボラティリティから推定
            beta = self.info.get('beta', 1.0)
            risk_free_rate = 0.03  # 3%と仮定（米国10年債利回りの近似値）
            market_risk_premium = 0.08  # 8%と仮定（歴史的な株式リスクプレミアム）

            cost_of_equity = risk_free_rate + beta * market_risk_premium

            # 負債コスト（Rd）の計算
            # Interest Expense / Total Debt
            if self.financials is not None and 'Interest Expense' in self.financials.index and total_debt > 0:
                interest_expense = abs(self.financials.loc['Interest Expense'].iloc[0])
                cost_of_debt = interest_expense / total_debt
            else:
                # 推定値を使用（投資適格社債の平均利回り）
                cost_of_debt = 0.04  # 4%と仮定

            # 実効税率
            if self.financials is not None and 'Tax Provision' in self.financials.index and 'Pretax Income' in self.financials.index:
                tax_provision = self.financials.loc['Tax Provision'].iloc[0]
                pretax_income = self.financials.loc['Pretax Income'].iloc[0]
                if pretax_income != 0:
                    tax_rate = abs(tax_provision / pretax_income)
                else:
                    tax_rate = 0.25
            else:
                tax_rate = 0.25  # デフォルト税率25%

            # WACCの計算
            wacc = (market_cap / enterprise_value * cost_of_equity +
                   total_debt / enterprise_value * cost_of_debt * (1 - tax_rate)) * 100

            return wacc

        except Exception as e:
            print(f"⚠️  WACC計算エラー: {e}")
            return None

    def get_company_info(self):
        """企業情報を取得"""
        if not self.info:
            return {}

        # ROICとWACCを計算
        roic = self.calculate_roic()
        wacc = self.calculate_wacc()

        info_dict = {
            '企業名': self.info.get('longName', 'N/A'),
            'セクター': self.info.get('sector', 'N/A'),
            '業種': self.info.get('industry', 'N/A'),
            '従業員数': self.info.get('fullTimeEmployees', 'N/A'),
            '時価総額': self._format_large_number(self.info.get('marketCap', 0)),
            'PER': round(self.info.get('trailingPE', 0), 2) if self.info.get('trailingPE') else 'N/A',
            'PBR': round(self.info.get('priceToBook', 0), 2) if self.info.get('priceToBook') else 'N/A',
            '配当利回り': f"{self.info.get('dividendYield', 0) * 100:.2f}%" if self.info.get('dividendYield') else 'N/A',
            'ROE': f"{self.info.get('returnOnEquity', 0) * 100:.2f}%" if self.info.get('returnOnEquity') else 'N/A',
            'ROIC': f"{roic:.2f}%" if roic is not None else 'N/A',
            'WACC': f"{wacc:.2f}%" if wacc is not None else 'N/A',
        }

        return info_dict

    def _format_large_number(self, num):
        """大きな数字を読みやすくフォーマット"""
        if num == 0 or num is None:
            return 'N/A'

        if num >= 1e12:
            return f"{num/1e12:.2f}兆"
        elif num >= 1e8:
            return f"{num/1e8:.2f}億"
        elif num >= 1e4:
            return f"{num/1e4:.2f}万"
        else:
            return f"{num:,.0f}"

    def print_company_info(self):
        """企業情報を表示"""
        info = self.get_company_info()
        print("\n" + "="*60)
        print(f"📈 企業情報: {info.get('企業名', self.ticker_code)}")
        print("="*60)
        for key, value in info.items():
            if key != '企業名':
                print(f"{key:12}: {value}")
        print("="*60 + "\n")

    def plot_analysis(self, save_path='financial_analysis.png'):
        """財務分析グラフを作成"""
        if self.historical_data is None or self.historical_data.empty:
            print("❌ データがありません")
            return

        # テクニカル指標を計算
        self.calculate_technical_indicators()

        # 図のサイズを設定
        fig = plt.figure(figsize=(20, 17))

        # グリッド設定
        gs = fig.add_gridspec(5, 2, hspace=0.3, wspace=0.3)

        company_name = self.info.get('longName', self.ticker_code) if self.info else self.ticker_code
        fig.suptitle(f'Financial Analysis: {company_name} ({self.ticker_symbol})',
                    fontsize=20, fontweight='bold', y=0.995)

        # 1. 株価チャート + 移動平均線
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(self.historical_data.index, self.historical_data['Close'],
                label='Close Price', linewidth=2, color='#1f77b4')
        ax1.plot(self.historical_data.index, self.historical_data['MA5'],
                label='MA5', linewidth=1.5, alpha=0.7, color='#ff7f0e')
        ax1.plot(self.historical_data.index, self.historical_data['MA25'],
                label='MA25', linewidth=1.5, alpha=0.7, color='#2ca02c')
        ax1.plot(self.historical_data.index, self.historical_data['MA75'],
                label='MA75', linewidth=1.5, alpha=0.7, color='#d62728')
        ax1.set_title('Stock Price & Moving Averages', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price', fontsize=12)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)

        # 2. ボリンジャーバンド
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(self.historical_data.index, self.historical_data['Close'],
                label='Close', linewidth=2, color='#1f77b4')
        ax2.plot(self.historical_data.index, self.historical_data['BB_Upper'],
                label='Upper Band', linewidth=1, alpha=0.7, color='#ff7f0e', linestyle='--')
        ax2.plot(self.historical_data.index, self.historical_data['BB_Middle'],
                label='Middle Band', linewidth=1, alpha=0.7, color='#2ca02c')
        ax2.plot(self.historical_data.index, self.historical_data['BB_Lower'],
                label='Lower Band', linewidth=1, alpha=0.7, color='#d62728', linestyle='--')
        ax2.fill_between(self.historical_data.index,
                         self.historical_data['BB_Upper'],
                         self.historical_data['BB_Lower'],
                         alpha=0.1, color='gray')
        ax2.set_title('Bollinger Bands', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Price', fontsize=12)
        ax2.legend(loc='best', fontsize=8)
        ax2.grid(True, alpha=0.3)

        # 3. 出来高
        ax3 = fig.add_subplot(gs[1, 1])
        colors = ['green' if close > open else 'red'
                 for close, open in zip(self.historical_data['Close'],
                                       self.historical_data['Open'])]
        ax3.bar(self.historical_data.index, self.historical_data['Volume'],
               color=colors, alpha=0.6, width=1)
        ax3.plot(self.historical_data.index, self.historical_data['Volume_MA'],
                label='Volume MA20', linewidth=2, color='blue')
        ax3.set_title('Trading Volume', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Volume', fontsize=12)
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)

        # 4. RSI
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.plot(self.historical_data.index, self.historical_data['RSI'],
                linewidth=2, color='purple')
        ax4.axhline(y=70, color='r', linestyle='--', linewidth=1, alpha=0.7, label='Overbought (70)')
        ax4.axhline(y=30, color='g', linestyle='--', linewidth=1, alpha=0.7, label='Oversold (30)')
        ax4.fill_between(self.historical_data.index, 30, 70, alpha=0.1, color='gray')
        ax4.set_title('RSI (Relative Strength Index)', fontsize=14, fontweight='bold')
        ax4.set_ylabel('RSI', fontsize=12)
        ax4.set_ylim(0, 100)
        ax4.legend(loc='best', fontsize=8)
        ax4.grid(True, alpha=0.3)

        # 5. MACD
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.plot(self.historical_data.index, self.historical_data['MACD'],
                label='MACD', linewidth=2, color='blue')
        ax5.plot(self.historical_data.index, self.historical_data['Signal'],
                label='Signal', linewidth=2, color='red')
        ax5.bar(self.historical_data.index,
               self.historical_data['MACD'] - self.historical_data['Signal'],
               label='Histogram', alpha=0.3, color='gray')
        ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax5.set_title('MACD', fontsize=14, fontweight='bold')
        ax5.set_ylabel('MACD', fontsize=12)
        ax5.legend(loc='best', fontsize=8)
        ax5.grid(True, alpha=0.3)

        # 6. 価格変動率（リターン分布）
        ax6 = fig.add_subplot(gs[3, 0])
        returns = self.historical_data['Close'].pct_change().dropna() * 100
        ax6.hist(returns, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax6.axvline(returns.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {returns.mean():.2f}%')
        ax6.axvline(returns.median(), color='green', linestyle='--', linewidth=2,
                   label=f'Median: {returns.median():.2f}%')
        ax6.set_title('Daily Returns Distribution', fontsize=14, fontweight='bold')
        ax6.set_xlabel('Daily Return (%)', fontsize=12)
        ax6.set_ylabel('Frequency', fontsize=12)
        ax6.legend(loc='best')
        ax6.grid(True, alpha=0.3)

        # 7. 価格統計情報
        ax7 = fig.add_subplot(gs[3, 1])
        ax7.axis('off')

        # 統計情報を計算
        current_price = self.historical_data['Close'].iloc[-1]
        price_change = current_price - self.historical_data['Close'].iloc[0]
        price_change_pct = (price_change / self.historical_data['Close'].iloc[0]) * 100

        stats_text = f"""
        Price Statistics:
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        Current Price:     {current_price:,.2f}
        Period Change:     {price_change:+,.2f} ({price_change_pct:+.2f}%)
        Period High:       {self.historical_data['High'].max():,.2f}
        Period Low:        {self.historical_data['Low'].min():,.2f}
        Average Volume:    {self.historical_data['Volume'].mean():,.0f}

        Volatility:        {returns.std():.2f}%
        Sharpe Ratio:      {returns.mean() / returns.std():.2f}
        Max Drawdown:      {((self.historical_data['Close'] / self.historical_data['Close'].cummax() - 1).min() * 100):.2f}%

        Last Updated:      {self.historical_data.index[-1].strftime('%Y-%m-%d')}
        """

        ax7.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        # 8. ROIC vs WACC 比較グラフ
        ax8 = fig.add_subplot(gs[4, 0])
        roic = self.calculate_roic()
        wacc = self.calculate_wacc()

        if roic is not None and wacc is not None:
            # バーグラフで表示
            metrics = ['ROIC', 'WACC']
            values = [roic, wacc]
            colors_bar = ['#2ecc71' if roic > wacc else '#e74c3c', '#3498db']

            bars = ax8.bar(metrics, values, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=2)

            # 値をバーの上に表示
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax8.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.2f}%',
                        ha='center', va='bottom', fontsize=12, fontweight='bold')

            # 基準線を追加
            ax8.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

            # 差分を表示
            spread = roic - wacc
            spread_text = f"Spread: {spread:+.2f}%"
            status_text = "Value Creating" if spread > 0 else "Value Destroying"
            status_color = '#2ecc71' if spread > 0 else '#e74c3c'

            ax8.text(0.5, 0.95, spread_text, transform=ax8.transAxes,
                    fontsize=12, ha='center', va='top',
                    bbox=dict(boxstyle='round', facecolor=status_color, alpha=0.3))
            ax8.text(0.5, 0.88, status_text, transform=ax8.transAxes,
                    fontsize=11, ha='center', va='top', fontweight='bold',
                    color=status_color)

            ax8.set_title('ROIC vs WACC Analysis', fontsize=14, fontweight='bold')
            ax8.set_ylabel('Rate (%)', fontsize=12)
            ax8.grid(True, alpha=0.3, axis='y')
        else:
            ax8.text(0.5, 0.5, 'ROIC/WACC data not available',
                    transform=ax8.transAxes, fontsize=12,
                    ha='center', va='center')
            ax8.set_title('ROIC vs WACC Analysis', fontsize=14, fontweight='bold')
            ax8.axis('off')

        # 9. 財務指標サマリー
        ax9 = fig.add_subplot(gs[4, 1])
        ax9.axis('off')

        # 財務指標を取得
        info = self.get_company_info()

        financial_text = f"""
        Financial Metrics Summary:
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        PER:               {info.get('PER', 'N/A')}
        PBR:               {info.get('PBR', 'N/A')}
        ROE:               {info.get('ROE', 'N/A')}
        ROIC:              {info.get('ROIC', 'N/A')}
        WACC:              {info.get('WACC', 'N/A')}

        Dividend Yield:    {info.get('配当利回り', 'N/A')}

        Market Cap:        {info.get('時価総額', 'N/A')}
        Sector:            {info.get('セクター', 'N/A')}

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        Enterprise Value Creation:
        ROIC > WACC = Value Creating ✓
        ROIC < WACC = Value Destroying ✗
        """

        ax9.text(0.1, 0.5, financial_text, fontsize=11, verticalalignment='center',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

        plt.tight_layout()

        # 保存
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ グラフを保存しました: {save_path}")

        return fig

    def plot_candlestick(self, save_path='candlestick_chart.png', days=90):
        """
        ローソク足チャートを作成

        Args:
            save_path (str): 保存先パス
            days (int): 表示する日数
        """
        if self.historical_data is None or self.historical_data.empty:
            print("❌ データがありません")
            return

        # 最近のデータを取得
        recent_data = self.historical_data.tail(days)

        # カラースタイルを設定
        mc = mpf.make_marketcolors(up='red', down='green', edge='inherit',
                                   wick='inherit', volume='in', alpha=0.8)
        s = mpf.make_mpf_style(marketcolors=mc, gridstyle='--', gridcolor='gray')

        # 移動平均線を追加
        apds = [
            mpf.make_addplot(recent_data['MA5'], color='orange', width=1.5),
            mpf.make_addplot(recent_data['MA25'], color='blue', width=1.5),
            mpf.make_addplot(recent_data['MA75'], color='purple', width=1.5),
        ]

        company_name = self.info.get('longName', self.ticker_code) if self.info else self.ticker_code

        # チャート作成
        fig, axes = mpf.plot(recent_data, type='candle', style=s,
                            volume=True, addplot=apds,
                            title=f'\n{company_name} ({self.ticker_symbol}) - Candlestick Chart',
                            ylabel='Price', ylabel_lower='Volume',
                            figsize=(16, 10), returnfig=True)

        # 保存
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ ローソク足チャートを保存しました: {save_path}")

        return fig


def main():
    """メイン関数"""
    print("\n" + "="*60)
    print("📊 財務分析ツール - Financial Analysis Tool")
    print("="*60 + "\n")

    # 証券コード入力
    if len(sys.argv) > 1:
        ticker_code = sys.argv[1]
    else:
        print("証券コードを入力してください")
        print("例: 日本株 -> 7203 (トヨタ), 6758 (ソニー)")
        print("例: 米国株 -> AAPL, GOOGL, MSFT")
        ticker_code = input("\n証券コード: ").strip()

    if not ticker_code:
        print("❌ 証券コードが入力されていません")
        return

    # 分析実行
    analyzer = FinancialAnalyzer(ticker_code)

    # データ取得
    if not analyzer.fetch_data(period="2y"):
        return

    # 企業情報表示
    analyzer.print_company_info()

    # グラフ作成
    print("📊 分析グラフを作成中...")
    analyzer.plot_analysis(save_path=f'{ticker_code}_financial_analysis.png')

    print("\n📊 ローソク足チャートを作成中...")
    analyzer.plot_candlestick(save_path=f'{ticker_code}_candlestick.png', days=90)

    print("\n" + "="*60)
    print("✅ 分析完了!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
