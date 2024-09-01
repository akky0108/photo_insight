import csv
import os

def process_details(file_path):
    try:
        # CSVファイルを開く
        with open(file_path, mode='r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)

            # 明細のリストを初期化
            details = []
            total_amount = 0.0

            # CSVファイルの各行を処理
            for row in csv_reader:
                item = row['Item']
                quantity = int(row['Quantity'])
                price = float(row['Price'])

                # 行の処理（合計金額の計算など）
                amount = quantity * price
                total_amount += amount

                # 明細リストに追加
                details.append({
                    'Item': item,
                    'Quantity': quantity,
                    'Price': price,
                    'Amount': amount
                })

            # 結果を出力
            print("Processed Details:")
            for detail in details:
                print(detail)

            print(f"\nTotal Amount: ${total_amount:.2f}")

    except FileNotFoundError:
        print(f"ファイルが見つかりません: {file_path}")
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")

if __name__ == "__main__":
    # 読み込むCSVファイルのパス
    project_root = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(project_root, 'data.csv')
    
    # 明細の処理
    process_details(file_path)
