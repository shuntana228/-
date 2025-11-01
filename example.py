#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
財務分析ツールの使用例
Example usage of Financial Analysis Tool
"""

from financial_analyzer import FinancialAnalyzer


def analyze_stock(ticker_code, period="2y"):
    """
    株式分析を実行

    Args:
        ticker_code (str): 証券コード
        period (str): 分析期間
    """
    print(f"\n{'='*60}")
    print(f"分析開始: {ticker_code}")
    print(f"{'='*60}\n")

    # アナライザーのインスタンス作成
    analyzer = FinancialAnalyzer(ticker_code)

    # データ取得
    if not analyzer.fetch_data(period=period):
        print(f"❌ {ticker_code} のデータ取得に失敗しました")
        return None

    # 企業情報を表示
    analyzer.print_company_info()

    # グラフを生成
    analyzer.plot_analysis(save_path=f'{ticker_code}_analysis.png')
    analyzer.plot_candlestick(save_path=f'{ticker_code}_candlestick.png', days=90)

    print(f"\n✅ {ticker_code} の分析が完了しました\n")

    return analyzer


def compare_multiple_stocks(ticker_codes):
    """
    複数の銘柄を分析

    Args:
        ticker_codes (list): 証券コードのリスト
    """
    results = {}

    for code in ticker_codes:
        analyzer = analyze_stock(code)
        if analyzer:
            results[code] = analyzer

    print(f"\n{'='*60}")
    print(f"✅ すべての分析が完了しました")
    print(f"分析した銘柄数: {len(results)}")
    print(f"{'='*60}\n")

    return results


# 使用例
if __name__ == "__main__":
    print("\n" + "="*60)
    print("📊 財務分析ツール - 使用例")
    print("="*60)

    # 例1: 単一銘柄の分析
    print("\n[例1] トヨタ自動車の分析")
    analyze_stock("7203", period="2y")

    # 例2: 複数銘柄の分析
    print("\n[例2] 複数銘柄の分析")
    japanese_stocks = ["7203", "6758", "9984"]  # トヨタ、ソニー、ソフトバンクG
    compare_multiple_stocks(japanese_stocks)

    # 例3: 米国株の分析
    print("\n[例3] 米国株の分析")
    us_stocks = ["AAPL", "GOOGL", "MSFT"]
    compare_multiple_stocks(us_stocks)

    print("\n" + "="*60)
    print("完了!")
    print("="*60 + "\n")
