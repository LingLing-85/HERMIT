import os
import glob
import pandas as pd
import csv
from pathlib import Path
from collections import Counter
from datetime import datetime, timedelta
import numpy as np
from tqdm import tqdm
import filelock

class CAIDADataProcessor:
    def __init__(self, input_folder="/mnt/kingston/merged_data/", output_folder="caida_dataset", reset_mappings=False):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.dataset_dir = output_folder
        
        # 映射表
        self.ip_to_id_map = {}
        self.time_to_id_map = {}
        self.next_ip_id = 1
        self.next_time_id = 1
        
        self.setup_directories()
        if reset_mappings:
            self.reset_mappings()
        else:
            self.load_mappings()

    def setup_directories(self):
        """建立必要的資料夾"""
        os.makedirs(self.dataset_dir, exist_ok=True)
        print(f"資料夾建立完成: {self.dataset_dir}")

    def reset_mappings(self):
        """強制重置所有映射，ID 從 1 開始"""
        mapping_files = [
            os.path.join(self.dataset_dir, "ip_to_id_mapping.csv"),
            os.path.join(self.dataset_dir, "time_to_id_mapping.csv")
        ]
        for file in mapping_files:
            if os.path.exists(file):
                os.remove(file)
                print(f"已刪除映射檔案: {file}")
        self.ip_to_id_map = {}
        self.time_to_id_map = {}
        self.next_ip_id = 1
        self.next_time_id = 1
        print("映射已重置，ID 將從 1 開始")

    def load_mappings(self):
        """載入現有的映射表"""
        self.load_ip_to_id_mapping()
        self.load_time_to_id_mapping()

    def get_or_create_ip_id(self, ip):
        """取得或建立 IP 的 ID"""
        ip = ip.strip()
        if ip not in self.ip_to_id_map:
            self.ip_to_id_map[ip] = self.next_ip_id
            self.next_ip_id += 1
            print(f"新增 IP ID: {ip} -> {self.ip_to_id_map[ip]}")
        else:
            print(f"重用 IP ID: {ip} -> {self.ip_to_id_map[ip]}")
        return self.ip_to_id_map[ip]

    def get_or_create_time_id(self, date):
        """取得或建立時間的 ID"""
        date = date.strip()
        if date not in self.time_to_id_map:
            self.time_to_id_map[date] = self.next_time_id
            self.next_time_id += 1
            print(f"新增時間 ID: {date} -> {self.time_to_id_map[date]}")
        else:
            print(f"重用時間 ID: {date} -> {self.time_to_id_map[date]}")
        return self.time_to_id_map[date]

    def save_ip_to_id_mapping(self):
        """儲存 IP 到 ID 的映射表"""
        mapping_file = os.path.join(self.dataset_dir, "ip_to_id_mapping.csv")
        with filelock.FileLock(mapping_file + ".lock"):
            with open(mapping_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['ip', 'id'])
                for ip, id_val in sorted(self.ip_to_id_map.items(), key=lambda x: x[1]):
                    writer.writerow([ip, id_val])
            print(f"IP到ID對應表已儲存至: {mapping_file}")

    def load_ip_to_id_mapping(self):
        """載入 IP 到 ID 的映射表"""
        mapping_file = os.path.join(self.dataset_dir, "ip_to_id_mapping.csv")
        if os.path.exists(mapping_file):
            try:
                df = pd.read_csv(mapping_file)
                duplicates = df[df['ip'].duplicated(keep=False)]
                if not duplicates.empty:
                    print(f"警告: 發現重複 IP 條目: {duplicates}")
                    df = df.drop_duplicates(subset=['ip'], keep='last')  # 保留最後一個重複條目
                self.ip_to_id_map = dict(zip(df['ip'].astype(str).str.strip(), df['id']))
                if self.ip_to_id_map:
                    self.next_ip_id = max(map(int, self.ip_to_id_map.values())) + 1
                else:
                    self.next_ip_id = 1
                print(f"載入現有IP映射: {len(self.ip_to_id_map)} 個IP，下個ID: {self.next_ip_id}")
            except Exception as e:
                print(f"載入IP對應表失敗: {e}")
                self.ip_to_id_map = {}
                self.next_ip_id = 1
        else:
            self.ip_to_id_map = {}
            self.next_ip_id = 1

    def save_time_to_id_mapping(self):
        """儲存時間到 ID 的映射表，按日期排序"""
        mapping_file = os.path.join(self.dataset_dir, "time_to_id_mapping.csv")
        with filelock.FileLock(mapping_file + ".lock"):
            with open(mapping_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['date', 'time_id'])
                for date, time_id in sorted(self.time_to_id_map.items(), key=lambda x: x[0]):
                    writer.writerow([date, time_id])
            print(f"時間到ID對應表已儲存至: {mapping_file}")

    def load_time_to_id_mapping(self):
        """載入時間到 ID 的映射表"""
        mapping_file = os.path.join(self.dataset_dir, "time_to_id_mapping.csv")
        if os.path.exists(mapping_file):
            try:
                df = pd.read_csv(mapping_file)
                # 檢查重複日期
                duplicates = df[df['date'].duplicated(keep=False)]
                if not duplicates.empty:
                    print(f"警告: 發現重複日期條目: {duplicates}")
                    df = df.drop_duplicates(subset=['date'], keep='last')  # 保留最後一個重複條目
                self.time_to_id_map = dict(zip(df['date'].astype(str).str.strip(), df['time_id']))
                if self.time_to_id_map:
                    self.next_time_id = max(map(int, self.time_to_id_map.values())) + 1
                else:
                    self.next_time_id = 1
                print(f"載入現有時間映射: {len(self.time_to_id_map)} 個日期，下個ID: {self.next_time_id}")
                print(f"'20150101' 的 ID: {self.time_to_id_map.get('20150101', '不存在')}")
            except Exception as e:
                print(f"載入時間對應表失敗: {e}")
                self.time_to_id_map = {}
                self.next_time_id = 1
        else:
            self.time_to_id_map = {}
            self.next_time_id = 1

    def extract_date_from_filename(self, filename):
        """從檔案名稱提取日期 (YYYYMMDD)"""
        try:
            basename = os.path.basename(filename).replace('.csv', '').strip()
            if len(basename) == 8 and basename.isdigit():
                return basename
            return None
        except Exception as e:
            print(f"提取日期失敗: {filename}, 錯誤: {e}")
            return None

    def find_csv_files_by_year(self, year):
        """找出指定年份的所有 CSV 檔案"""
        pattern = os.path.join(self.input_folder, f"{year}*.csv")
        csv_files = sorted(glob.glob(pattern))
        valid_files = []
        for csv_file in csv_files:
            date = self.extract_date_from_filename(csv_file)
            if date and date.startswith(str(year)):
                valid_files.append(csv_file)
        return valid_files

    def find_csv_files_by_month(self, year, month):
        """找出指定年月的所有 CSV 檔案"""
        month_str = f"{year}{month:02d}"
        pattern = os.path.join(self.input_folder, f"{month_str}*.csv")
        csv_files = sorted(glob.glob(pattern))
        valid_files = []
        for csv_file in csv_files:
            date = self.extract_date_from_filename(csv_file)
            if date and date.startswith(month_str):
                valid_files.append(csv_file)
        return valid_files

    def process_single_day_file(self, csv_file):
        """處理單一天的 CSV 檔案，返回處理後的資料"""
        try:
            date = self.extract_date_from_filename(csv_file)
            if not date:
                print(f"無法提取日期: {csv_file}")
                return None
            print(f"處理檔案: {os.path.basename(csv_file)} (日期: {date})")
            df = pd.read_csv(csv_file)
            df.columns = df.columns.str.strip()
            required_columns = ['source', 'target']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"缺少必要欄位: {missing_columns}")
                return None
            if len(df) == 0:
                print(f"檔案為空: {csv_file}")
                return None
            df['source'] = df['source'].astype(str).str.strip()
            df['target'] = df['target'].astype(str).str.strip()
            df = df.dropna(subset=['source', 'target'])
            df = df[(df['source'] != '') & (df['target'] != '')]
            if len(df) == 0:
                print(f"清理後沒有有效資料: {csv_file}")
                return None
            print(f"  原始邊數: {len(df)}")
            edge_weights = df.groupby(['source', 'target']).size().reset_index(name='weight')
            print(f"  去重後邊數: {len(edge_weights)}")
            edge_weights['date'] = date
            return edge_weights
        except Exception as e:
            print(f"處理檔案 {csv_file} 時發生錯誤: {e}")
            return None

    def apply_top_nodes_filtering(self, merged_df):
        """應用 top 20% 節點過濾，並生成 induced subgraph"""
        print("計算節點度數...")
        node_weights = {}
        for _, row in merged_df.iterrows():
            src, dst, weight = row['source'], row['target'], row['weight']
            node_weights[src] = node_weights.get(src, 0) + weight
            node_weights[dst] = node_weights.get(dst, 0) + weight
        total_nodes = len(node_weights)
        top_n = max(1, int(total_nodes * 0.1)) # 10%
        top_nodes = sorted(node_weights.items(), key=lambda x: x[1], reverse=True)[:top_n]
        top_node_list = [node for node, _ in top_nodes]
        print(f"總節點數: {total_nodes}")
        print(f"選擇前 {top_n} 個節點 (10%)")
        top_node_total_weight = sum(weight for _, weight in top_nodes)
        total_weight = sum(node_weights.values())
        coverage_percentage = (top_node_total_weight / total_weight) * 100
        print(f"前 10% 節點權重佔比: {coverage_percentage:.1f}%")
        print("生成 induced subgraph...")
        filtered_edges = merged_df[
            merged_df['source'].isin(top_node_list) &
            merged_df['target'].isin(top_node_list)
        ].copy()
        if len(filtered_edges) == 0:
            print("過濾後沒有邊")
            return None
        print(f"過濾後邊數: {len(filtered_edges)}")
        return filtered_edges

    def convert_to_final_format(self, filtered_df):
        """轉換為最終格式 (source, target, weight, time)"""
        print("轉換為最終格式...")
        print("轉換 source IP 為 ID...")
        filtered_df['source'] = filtered_df['source'].apply(self.get_or_create_ip_id)
        print("轉換 target IP 為 ID...")
        filtered_df['target'] = filtered_df['target'].apply(self.get_or_create_ip_id)
        print("轉換日期為時間 ID...")
        filtered_df['time'] = filtered_df['date'].apply(self.get_or_create_time_id)
        final_df = filtered_df[['source', 'target', 'weight', 'time']].copy()
        return final_df

    def process_year_data(self, year, apply_filtering=True):
        """處理指定年份的所有資料"""
        print(f"\n=== 處理 {year} 年資料 ===")
        csv_files = self.find_csv_files_by_year(year)
        if not csv_files:
            print(f"在 {self.input_folder} 中沒有找到 {year} 年的 CSV 檔案")
            return
        print(f"找到 {len(csv_files)} 個 {year} 年的 CSV 檔案")
        all_dataframes = []
        for csv_file in tqdm(csv_files, desc=f"處理 {year} 年檔案"):
            day_data = self.process_single_day_file(csv_file)
            if day_data is not None:
                all_dataframes.append(day_data)
        if not all_dataframes:
            print(f"沒有成功處理到任何 {year} 年的檔案")
            return
        print(f"\n正在合併 {len(all_dataframes)} 個檔案...")
        merged_df = pd.concat(all_dataframes, ignore_index=True)
        print(f"合併後總邊數: {len(merged_df)}")
        print(f"唯一日期數: {merged_df['date'].nunique()}")
        if apply_filtering:
            filtered_df = self.apply_top_nodes_filtering(merged_df)
            if filtered_df is None:
                return
        else:
            print("跳過節點過濾，保留所有節點")
            filtered_df = merged_df
        final_df = self.convert_to_final_format(filtered_df)
        output_file = os.path.join(self.dataset_dir, f"{year}_all_10percent.csv")
        final_df.to_csv(output_file, index=False)
        print(f"\n最終結果已儲存: {output_file}")
        print(f"最終統計: {len(final_df)} 條邊")
        print(f"節點數: {len(set(final_df['source']) | set(final_df['target']))}")
        print(f"時間範圍: {final_df['time'].min()} - {final_df['time'].max()}")
        self.save_ip_to_id_mapping()
        self.save_time_to_id_mapping()
        return output_file

    def process_month_data(self, year, month, apply_filtering=True):
        """處理指定年月的資料"""
        print(f"\n=== 處理 {year} 年 {month} 月資料 ===")
        csv_files = self.find_csv_files_by_month(year, month)
        if not csv_files:
            print(f"在 {self.input_folder} 中沒有找到 {year} 年 {month} 月的 CSV 檔案")
            return
        print(f"找到 {len(csv_files)} 個檔案")
        print("找到的檔案:")
        for csv_file in csv_files:
            print(f"  - {os.path.basename(csv_file)}")
        all_dataframes = []
        for csv_file in tqdm(csv_files, desc=f"處理 {year}/{month:02d}"):
            day_data = self.process_single_day_file(csv_file)
            if day_data is not None:
                all_dataframes.append(day_data)
        if not all_dataframes:
            print(f"沒有成功處理到任何檔案")
            return
        print(f"\n正在合併所有檔案...")
        merged_df = pd.concat(all_dataframes, ignore_index=True)
        print(f"合併後總邊數: {len(merged_df)}")
        if apply_filtering:
            filtered_df = self.apply_top_nodes_filtering(merged_df)
            if filtered_df is None:
                return
        else:
            print("跳過節點過濾，保留所有節點")
            filtered_df = merged_df
        final_df = self.convert_to_final_format(filtered_df)
        output_file = os.path.join(self.dataset_dir, f"{year}{month:02d}_all.csv")
        final_df.to_csv(output_file, index=False)
        print(f"\n最終結果已儲存: {output_file}")
        print(f"最終統計: {len(final_df)} 條邊")
        self.save_ip_to_id_mapping()
        self.save_time_to_id_mapping()
        return output_file

    def process_multiple_years_data(self, start_year, end_year, apply_filtering=True):
        """處理多個年份的資料"""
        for year in range(start_year, end_year + 1):
            self.process_year_data(year, apply_filtering=apply_filtering)

    def show_statistics(self, year=None):
        """顯示指定年份或所有年份的統計資訊"""
        if year:
            pattern = os.path.join(self.input_folder, f"{year}*.csv")
        else:
            pattern = os.path.join(self.input_folder, "*.csv")
        csv_files = sorted(glob.glob(pattern))
        if not csv_files:
            print("沒有找到任何 CSV 檔案")
            return
        print(f"\n=== 統計資訊 {'('+str(year)+' 年)' if year else ''} ===")
        print(f"找到 {len(csv_files)} 個 CSV 檔案")
        year_counts = {}
        for csv_file in csv_files:
            date = self.extract_date_from_filename(csv_file)
            if date and len(date) >= 4:
                file_year = date[:4]
                year_counts[file_year] = year_counts.get(file_year, 0) + 1
        print("\n每年檔案數量:")
        for y in sorted(year_counts.keys()):
            print(f"  {y}: {year_counts[y]} 個檔案")
        dates = []
        for csv_file in csv_files:
            date = self.extract_date_from_filename(csv_file)
            if date:
                dates.append(date)
        if dates:
            dates.sort()
            print(f"\n日期範圍: {dates[0]} - {dates[-1]}")
            print(f"總共 {len(dates)} 天的資料")

def main():
    input_folder = "/mnt/kingston/merged_data/"
    output_folder = "/mnt/NewSSD/merged_data_with_time_weight/"
    processor = CAIDADataProcessor(input_folder=input_folder, output_folder=output_folder, reset_mappings=False)
    
    print("CAIDA 資料處理工具")
    print("="*50)
    print("1. 處理單一年份 (包含 top 10% 節點過濾)")
    print("2. 處理單一月份 (包含 top 10% 節點過濾)")
    print("3. 處理單一年份 (不過濾節點)")
    print("4. 顯示資料統計")
    print("5. 直接處理 2015 年資料")
    print("6. 處理多個年份 (包含 top 10% 節點過濾)")
    
    choice = input("\n請選擇 (1-6): ").strip()
    
    if choice == "1":
        year = input("請輸入年份 (例如: 2015): ").strip()
        try:
            year = int(year)
            processor.process_year_data(year, apply_filtering=True)
        except ValueError:
            print("無效的年份格式")
    
    elif choice == "2":
        year = input("請輸入年份 (例如: 2015): ").strip()
        month = input("請輸入月份 (例如: 1): ").strip()
        try:
            year = int(year)
            month = int(month)
            if 1 <= month <= 12:
                processor.process_month_data(year, month, apply_filtering=True)
            else:
                print("月份必須在 1-12 之間")
        except ValueError:
            print("無效的年份或月份格式")
    
    elif choice == "3":
        year = input("請輸入年份 (例如: 2015): ").strip()
        try:
            year = int(year)
            processor.process_year_data(year, apply_filtering=False)
        except ValueError:
            print("無效的年份格式")
    
    elif choice == "4":
        year_input = input("請輸入年份 (留空顯示所有年份): ").strip()
        if year_input:
            try:
                year = int(year_input)
                processor.show_statistics(year)
            except ValueError:
                print("無效的年份格式")
        else:
            processor.show_statistics()
    
    elif choice == "5":
        processor.process_year_data(2015, apply_filtering=True)
    
    elif choice == "6":
        start_year = input("請輸入開始年份 (例如: 2015): ").strip()
        end_year = input("請輸入結束年份 (例如: 2020): ").strip()
        try:
            start_year = int(start_year)
            end_year = int(end_year)
            if start_year <= end_year:
                processor.process_multiple_years_data(start_year, end_year, apply_filtering=True)
            else:
                print("開始年份必須小於或等於結束年份")
        except ValueError:
            print("無效的年份格式")
    
    else:
        print("無效的選擇")

if __name__ == "__main__":
    main()