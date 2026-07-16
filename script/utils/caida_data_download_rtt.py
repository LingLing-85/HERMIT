import os
import csv
import json
import gzip
import shutil
import subprocess
import requests
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from tqdm import tqdm
import pandas as pd
import numpy as np
from collections import Counter
import re

class CAIDADataCollector:
    def __init__(self):
        self.base_url = "https://publicdata.caida.org/datasets/topology/ark/ipv4/probe-data/team-1"
        self.temp_dir = "temp"
        self.processed_dir = "processed_data"
        self.merged_dir = "merged_data"
        self.dataset_dir = "caida_dataset"
        self.ip_to_id_map = {}
        self.time_to_id_map = {}
        self.next_ip_id = 1
        self.next_time_id = 1
        self.setup_directories()
        self.load_mappings()

    def setup_directories(self):
        for dir_path in [self.temp_dir, self.processed_dir, self.merged_dir, self.dataset_dir]:
            os.makedirs(dir_path, exist_ok=True)
        print("資料夾建立完成")

    def load_mappings(self):
        self.load_ip_to_id_mapping()
        self.load_time_to_id_mapping()

    def get_or_create_ip_id(self, ip):
        if ip not in self.ip_to_id_map:
            self.ip_to_id_map[ip] = self.next_ip_id
            self.next_ip_id += 1
        return self.ip_to_id_map[ip]

    def get_or_create_time_id(self, date):
        if date not in self.time_to_id_map:
            self.time_to_id_map[date] = self.next_time_id
            self.next_time_id += 1
        return self.time_to_id_map[date]

    def save_ip_to_id_mapping(self):
        mapping_file = os.path.join(self.dataset_dir, "ip_to_id_mapping.csv")
        with open(mapping_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['ip', 'id'])
            for ip, id_val in sorted(self.ip_to_id_map.items(), key=lambda x: x[1]):
                writer.writerow([ip, id_val])
        print(f"IP到ID對應表已儲存至: {mapping_file}")

    def load_ip_to_id_mapping(self):
        mapping_file = os.path.join(self.dataset_dir, "ip_to_id_mapping.csv")
        if os.path.exists(mapping_file):
            try:
                df = pd.read_csv(mapping_file)
                self.ip_to_id_map = dict(zip(df['ip'], df['id']))
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
        mapping_file = os.path.join(self.dataset_dir, "time_to_id_mapping.csv")
        with open(mapping_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['date', 'time_id'])
            for date, time_id in sorted(self.time_to_id_map.items(), key=lambda x: x[1]):
                writer.writerow([date, time_id])
        print(f"時間到ID對應表已儲存至: {mapping_file}")

    def load_time_to_id_mapping(self):
        mapping_file = os.path.join(self.dataset_dir, "time_to_id_mapping.csv")
        if os.path.exists(mapping_file):
            try:
                df = pd.read_csv(mapping_file)
                self.time_to_id_map = dict(zip(df['date'], df['time_id']))
                if self.time_to_id_map:
                    self.next_time_id = max(map(int, self.time_to_id_map.values())) + 1
                else:
                    self.next_time_id = 1
                print(f"載入現有時間映射: {len(self.time_to_id_map)} 個日期，下個ID: {self.next_time_id}")
            except Exception as e:
                print(f"載入時間對應表失敗: {e}")
                self.time_to_id_map = {}
                self.next_time_id = 1
        else:
            self.time_to_id_map = {}
            self.next_time_id = 1

    def is_valid_ip(self, ip):
        try:
            parts = ip.split('.')
            if len(parts) != 4:
                return False
            for part in parts:
                if not (0 <= int(part) <= 255):
                    return False
            return True
        except:
            return False

    def to_datetime(self, t):
        return datetime.utcfromtimestamp(t["sec"]) + timedelta(microseconds=t["usec"])

    def extract_state_from_filename(self, filename):
        try:
            parts = filename.split('.')
            if len(parts) >= 6:
                state_country = parts[5]
                if '-' in state_country and state_country.endswith('-us'):
                    state = state_country[:-3]
                    return state
            return "unknown"
        except Exception as e:
            print(f"提取州名失敗: {filename}, 錯誤: {e}")
            return "unknown"

    def extract_date_from_filename(self, filename):
        try:
            parts = filename.split('.')
            if len(parts) >= 5:
                date_part = parts[4]
                if len(date_part) == 8 and date_part.isdigit():
                    return date_part
            return None
        except Exception as e:
            print(f"提取日期失敗: {filename}, 錯誤: {e}")
            return None

    def convert_trace_to_edges(self, trace):
        """將trace轉換為邊列表，包含RTT"""
        edges = []
        hops = trace.get("hops", [])
        
        if len(hops) < 2:
            return edges
            
        for i in range(len(hops) - 1):
            src = hops[i].get("addr")
            dst = hops[i+1].get("addr")
            rtt = hops[i+1].get("rtt")
            if not src or not dst or rtt is None:
                continue
            if not self.is_valid_ip(src) or not self.is_valid_ip(dst):
                continue
            if not isinstance(rtt, (int, float)) or rtt <= 0:
                continue
            edges.append((src, dst, rtt))
        return edges

    def get_candidate_dates(self, target_date):
        target_dt = datetime.strptime(target_date, "%Y%m%d")
        candidate_dates = []
        for offset in [-1, 0, 1]:
            candidate_dt = target_dt + timedelta(days=offset)
            candidate_dates.append(candidate_dt.strftime("%Y%m%d"))
        return candidate_dates

    def find_files_for_date(self, target_date):
        candidate_dates = self.get_candidate_dates(target_date)
        all_files = []
        print(f"搜尋日期 {target_date} 的檔案，檢查候選日期: {candidate_dates}")
        
        for search_date in candidate_dates:
            year = search_date[:4]
            cycle_url = f"{self.base_url}/{year}/cycle-{search_date}/"
            
            try:
                resp = requests.get(cycle_url, timeout=30)
                if resp.status_code != 200:
                    print(f"無法取得資料頁面 {search_date}: {resp.status_code}")
                    continue
                soup = BeautifulSoup(resp.text, "html.parser")
                for a in soup.find_all('a', href=True):
                    href = a.get('href')
                    if isinstance(href, str) and '-us.warts.gz' in href:
                        file_date = self.extract_date_from_filename(href)
                        if file_date == target_date:
                            all_files.append({
                                'filename': href,
                                'search_date': search_date,
                                'file_date': file_date,
                                'state': self.extract_state_from_filename(href)
                            })
                            print(f"  找到: {href} (在 {search_date} 資料夾)")
            except Exception as e:
                print(f"搜尋日期 {search_date} 時發生錯誤: {e}")
                continue
        print(f"目標日期 {target_date} 總共找到 {len(all_files)} 個檔案")
        return all_files

    def download_and_convert_to_json(self, year, search_date, filename):
        url = f"{self.base_url}/{year}/cycle-{search_date}/{filename}"
        local_gz = os.path.join(self.temp_dir, filename)
        local_warts = local_gz[:-3]
        output_json = local_warts + ".json"
        try:
            print(f"下載: {filename} (從 {search_date} 資料夾)")
            r = requests.get(url, stream=True, timeout=60)
            if r.status_code != 200:
                print(f"下載失敗: {r.status_code} {url}")
                return None
            with open(local_gz, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            with gzip.open(local_gz, 'rb') as gz_file:
                with open(local_warts, 'wb') as warts_file:
                    shutil.copyfileobj(gz_file, warts_file)
            with open(output_json, 'w') as out_json:
                result = subprocess.run(["sc_warts2json", local_warts], 
                                      stdout=out_json, stderr=subprocess.PIPE)
            if result.returncode != 0:
                print(f"sc_warts2json 錯誤: {result.stderr.decode()}")
                return None
            for f in [local_gz, local_warts]:
                if os.path.exists(f):
                    os.remove(f)
            return output_json if os.path.exists(output_json) and os.path.getsize(output_json) > 0 else None
        except Exception as e:
            print(f"處理檔案 {filename} 時發生錯誤: {e}")
            return None

    def process_single_state_file(self, json_file, state, date):
        output_file = os.path.join(self.processed_dir, f"{state}-{date}.csv")
        if os.path.exists(output_file):
            print(f"檔案已存在，跳過: {output_file}")
            return output_file
        edges_count = 0
        completed_traces = 0
        total_traces = 0
        try:
            with open(output_file, 'w', newline='') as out_csv:
                writer = csv.writer(out_csv)
                writer.writerow(['source', 'target', 'rtt'])
                with open(json_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            if data.get("type") != "trace":
                                continue
                            total_traces += 1
                            if data.get("stop_reason") != "COMPLETED":
                                continue
                            completed_traces += 1
                            edges = self.convert_trace_to_edges(data)
                            if edges:
                                writer.writerows(edges)
                                edges_count += len(edges)
                        except json.JSONDecodeError:
                            continue
            print(f"州 {state} 日期 {date}: {total_traces} traces, {completed_traces} completed, {edges_count} edges")
            return output_file if edges_count > 0 else None
        except Exception as e:
            print(f"處理州檔案時發生錯誤: {e}")
            return None

    def merge_daily_data(self, date, state_files):
        merged_file = os.path.join(self.merged_dir, f"{date}.csv")
        if os.path.exists(merged_file):
            print(f"合併檔案已存在，跳過: {merged_file}")
            return merged_file
        total_edges = 0
        try:
            with open(merged_file, 'w', newline='') as out_csv:
                writer = csv.writer(out_csv)
                writer.writerow(['source', 'target', 'rtt'])
                for state_file in state_files:
                    if not os.path.exists(state_file):
                        continue
                    with open(state_file, 'r') as f:
                        reader = csv.reader(f)
                        next(reader)
                        for row in reader:
                            if len(row) == 3:
                                writer.writerow(row)
                                total_edges += 1
            print(f"日期 {date} 合併完成: {total_edges} 條邊")
            return merged_file if total_edges > 0 else None
        except Exception as e:
            print(f"合併日期 {date} 資料時發生錯誤: {e}")
            return None

    def process_final_dataset(self, date, merged_file):
        print(f"處理最終資料集: {date}")
        try:
            df = pd.read_csv(merged_file)
            print(f"原始邊數: {len(df)}")
            if len(df) == 0:
                print("沒有資料可處理")
                return
            df['source'] = df['source'].astype(str).str.strip()
            df['target'] = df['target'].astype(str).str.strip()
            df['rtt'] = df['rtt'].astype(float)
            print("計算邊權重和RTT統計...")
            edge_stats = df.groupby(['source', 'target']).agg({
                'rtt': ['count', 'mean', 'std']
            }).reset_index()
            edge_stats.columns = ['source', 'target', 'weight', 'avg_rtt', 'std_rtt']
            edge_stats['std_rtt'] = edge_stats.apply(
                lambda row: 0 if row['weight'] == 1 else row['std_rtt'],
                axis=1
            )
            edge_stats['std_rtt'] = edge_stats['std_rtt'].fillna(0)
            print(f"去重後邊數: {len(edge_stats)}")
            print("計算節點度數...")
            node_weights = {}
            for _, row in edge_stats.iterrows():
                src, dst, weight = row['source'], row['target'], row['weight']
                node_weights[src] = node_weights.get(src, 0) + weight
                node_weights[dst] = node_weights.get(dst, 0) + weight
            total_nodes = len(node_weights)
            top_n = max(1, int(total_nodes * 0.05))
            top_nodes = sorted(node_weights.items(), key=lambda x: x[1], reverse=True)[:top_n]
            top_node_list = [node for node, _ in top_nodes]
            print(f"總節點數: {total_nodes}")
            print(f"選擇前 {top_n} 個節點 (5%)")
            print("生成induced subgraph...")
            filtered_edges = edge_stats[
                edge_stats['source'].isin(top_node_list) &
                edge_stats['target'].isin(top_node_list)
            ].copy()
            if len(filtered_edges) == 0:
                print("過濾後沒有邊")
                return
            print("轉換為ID...")
            filtered_edges['source'] = filtered_edges['source'].apply(self.get_or_create_ip_id)
            filtered_edges['target'] = filtered_edges['target'].apply(self.get_or_create_ip_id)
            filtered_edges['time'] = self.get_or_create_time_id(date)
            filtered_edges = filtered_edges[['source', 'target', 'time', 'weight', 'avg_rtt', 'std_rtt']]
            output_file = os.path.join(self.dataset_dir, f"{date}.csv")
            filtered_edges.to_csv(output_file, index=False)
            print(f"最終結果已儲存: {output_file}")
            print(f"最終統計: {len(filtered_edges)} 條邊")
        except Exception as e:
            print(f"處理最終資料集時發生錯誤: {e}")
            import traceback
            traceback.print_exc()

    def generate_date_range(self, start_date, end_date):
        start = datetime.strptime(start_date, "%Y%m%d")
        end = datetime.strptime(end_date, "%Y%m%d")
        dates = []
        current = start
        while current <= end:
            dates.append(current.strftime("%Y%m%d"))
            current += timedelta(days=1)
        return dates

    def scan_and_index_files(self, start_date, end_date):
        date_range = self.generate_date_range(start_date, end_date)
        all_search_dates = set()
        for date in date_range:
            candidate_dates = self.get_candidate_dates(date)
            all_search_dates.update(candidate_dates)
        print(f"需要掃描的資料夾日期: {sorted(all_search_dates)}")
        file_index = {}
        for search_date in sorted(all_search_dates):
            print(f"掃描資料夾: {search_date}")
            year = search_date[:4]
            cycle_url = f"{self.base_url}/{year}/cycle-{search_date}/"
            try:
                resp = requests.get(cycle_url, timeout=30)
                if resp.status_code != 200:
                    print(f"  無法訪問: {resp.status_code}")
                    continue
                soup = BeautifulSoup(resp.text, "html.parser")
                folder_files = []
                for a in soup.find_all('a', href=True):
                    href = a.get('href')
                    if isinstance(href, str) and '-us.warts.gz' in href:
                        file_date = self.extract_date_from_filename(href)
                        if file_date:
                            folder_files.append({
                                'filename': href,
                                'file_date': file_date,
                                'state': self.extract_state_from_filename(href)
                            })
                if folder_files:
                    file_index[search_date] = folder_files
                    print(f"  找到 {len(folder_files)} 個檔案")
                else:
                    print(f"  沒有找到檔案")
            except Exception as e:
                print(f"  掃描失敗: {e}")
                continue
        return file_index

    def plan_downloads(self, file_index, target_dates):
        download_plan = {}
        for target_date in target_dates:
            print(f"\n=== 規劃日期: {target_date} ===")
            target_files = []
            for search_date, files in file_index.items():
                for file_info in files:
                    if file_info['file_date'] == target_date:
                        target_files.append({
                            'filename': file_info['filename'],
                            'search_date': search_date,
                            'file_date': file_info['file_date'],
                            'state': file_info['state']
                        })
            if target_files:
                download_plan[target_date] = target_files
                state_counts = {}
                for file_info in target_files:
                    state = file_info['state']
                    state_counts[state] = state_counts.get(state, 0) + 1
                print(f"找到 {len(target_files)} 個檔案")
                print(f"州分佈: {dict(sorted(state_counts.items()))}")
                folder_dist = {}
                for file_info in target_files:
                    folder = file_info['search_date']
                    folder_dist[folder] = folder_dist.get(folder, 0) + 1
                if len(folder_dist) > 1:
                    print(f"檔案分佈在多個資料夾: {folder_dist}")
                else:
                    print(f"檔案都在資料夾: {list(folder_dist.keys())[0]}")
            else:
                print(f"沒有找到任何檔案")
        return download_plan

    def execute_download_plan(self, download_plan):
        for date, file_list in download_plan.items():
            print(f"\n=== 執行下載: {date} ===")
            try:
                state_files = []
                for file_info in tqdm(file_list, desc=f"下載 {date}"):
                    filename = file_info['filename']
                    search_date = file_info['search_date']
                    state = file_info['state']
                    processed_file = os.path.join(self.processed_dir, f"{state}-{date}.csv")
                    if os.path.exists(processed_file):
                        print(f"已存在: {state}-{date}.csv")
                        state_files.append(processed_file)
                        continue
                    year = search_date[:4]
                    json_file = self.download_and_convert_to_json(year, search_date, filename)
                    if not json_file:
                        print(f"下載失敗: {filename}")
                        continue
                    processed_file = self.process_single_state_file(json_file, state, date)
                    if processed_file:
                        state_files.append(processed_file)
                        print(f"處理完成: {state}-{date}.csv")
                    if os.path.exists(json_file):
                        os.remove(json_file)
                print(f"成功處理的州檔案數量: {len(state_files)}")
                if state_files:
                    merged_file = self.merge_daily_data(date, state_files)
                    if merged_file:
                        self.process_final_dataset(date, merged_file)
                else:
                    print(f"日期 {date} 沒有成功處理的州資料")
            except Exception as e:
                print(f"處理日期 {date} 時發生錯誤: {e}")
                import traceback
                traceback.print_exc()
                continue

    def print_download_summary(self, download_plan):
        print("\n" + "="*60)
        print("下載計畫摘要")
        print("="*60)
        total_files = 0
        total_days = len(download_plan)
        for date, file_list in sorted(download_plan.items()):
            total_files += len(file_list)
            state_counts = {}
            folder_dist = {}
            for file_info in file_list:
                state = file_info['state']
                folder = file_info['search_date']
                state_counts[state] = state_counts.get(state, 0) + 1
                folder_dist[folder] = folder_dist.get(folder, 0) + 1
            print(f"\n日期: {date}")
            print(f"   檔案數: {len(file_list)}")
            print(f"   州數: {len(state_counts)}")
            print(f"   資料夾分佈: {dict(sorted(folder_dist.items()))}")
            if len(folder_dist) > 1:
                print(f"注意：檔案分散在 {len(folder_dist)} 個資料夾")
        print(f"\n總計:")
        print(f"   處理天數: {total_days}")
        print(f"   總檔案數: {total_files}")
        print(f"   估計下載大小: ~{total_files * 50}MB (假設每檔案約50MB)")
        print("="*60)

    def download_and_process(self, start_date, end_date, auto_confirm=False):
        date_range = self.generate_date_range(start_date, end_date)
        print("=== 第一階段：掃描和索引檔案 ===")
        file_index = self.scan_and_index_files(start_date, end_date)
        print("\n=== 第二階段：規劃下載 ===")
        download_plan = self.plan_downloads(file_index, date_range)
        self.print_download_summary(download_plan)
        if not auto_confirm:
            confirm = input("\n是否繼續執行下載？(y/N): ").strip().lower()
            if confirm not in ['y', 'yes', '是']:
                print("取消下載")
                return
        print("\n=== 第三階段：執行下載和處理 ===")
        self.execute_download_plan(download_plan)
        self.save_ip_to_id_mapping()
        self.save_time_to_id_mapping()
        print("\n=== 處理完成 ===")

    def scan_only(self, start_date, end_date):
        date_range = self.generate_date_range(start_date, end_date)
        print("=== 掃描模式：僅檢查可用檔案 ===")
        file_index = self.scan_and_index_files(start_date, end_date)
        print("\n=== 規劃下載 ===")
        download_plan = self.plan_downloads(file_index, date_range)
        self.print_download_summary(download_plan)
        return download_plan

if __name__ == "__main__":
    collector = CAIDADataCollector()
    collector.download_and_process(
        start_date="20180101",
        end_date="20180829",
        auto_confirm=True
    )