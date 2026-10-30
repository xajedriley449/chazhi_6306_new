import pandas as pd
import re
import numpy as np
from pathlib import Path

ASSUME_UNSIGNED_IS_AHEAD = False


def extract_robust_face_distance(row):
    """从“位置”和“巷道”解析有符号工作面距离，单位为米。"""
    pos_text = str(row["位置"]) if pd.notna(row["位置"]) else ""
    tun_text = str(row["巷道"]) if pd.notna(row["巷道"]) else ""
    combined_text = f"{pos_text} {tun_text}".strip()

    if not combined_text or combined_text.lower() in {"nan", "nan nan"}:
        return np.nan

    # 显式方向词优先；文字方向决定符号，数值部分按距离大小处理。
    explicit_match = re.search(
        r"(面前|面后)\s*([+-]?\d+(?:\.\d+)?)\s*米?",
        combined_text,
    )
    if explicit_match:
        direction = explicit_match.group(1)
        magnitude = abs(float(explicit_match.group(2)))
        return magnitude if direction == "面前" else -magnitude

    # 去除工作面编号、巷道名称等容易被误识别为距离的内容。
    noise_pattern = r"(730\d|530\d|2#|1#|七采区|工作面|轨顺|皮顺|回风|辅运|探巷|东翼|西翼)"
    clean_text = re.sub(noise_pattern, " ", combined_text)

    # 保留显式正负号；例如 -698 必须解析为 -698，而不是 +698。
    implicit_match = re.search(r"([+-]?\d+(?:\.\d+)?)\s*米?", clean_text)
    if not implicit_match:
        return np.nan

    token = implicit_match.group(1)
    value = float(token)

    # 有符号数字本身提供了方向信息。
    if token.startswith(("+", "-")):
        return value

    # 无符号数字没有方向信息，只能依据数据录入规范决定是否推断为面前。
    if ASSUME_UNSIGNED_IS_AHEAD:
        return value
    return np.nan



def main():
    script_dir = Path(__file__).resolve().parent
    all_csv_path = script_dir / '7303_all.csv'
    rotated_csv_path = script_dir / '7303_xyzet_rotated.csv'
    
    # 1. 读取原始数据 (注意编码)
    print("读取原始表...")
    try:
        df_all = pd.read_csv(all_csv_path, encoding='gbk')
    except UnicodeDecodeError:
        df_all = pd.read_csv(all_csv_path, encoding='utf-8')
        
    # 2. 读取旋转坐标表
    print("读取坐标表...")
    df_rotated = pd.read_csv(rotated_csv_path)
    
    # 3. 使用 apply(..., axis=1) 跨列联合解析
    print("正在联合解析 '位置' 和 '巷道'...")
    # 注意这里传入的是 df_all，让函数能同时访问每一行的多个列
    extracted_distances = df_all.apply(extract_robust_face_distance, axis=1)
    
    # 4. 追加到新表
    df_new = df_rotated.copy()
    df_new['dist_to_face_m'] = extracted_distances
    
    print("\n========= 提取结果统计 =========")
    print(f"总计数据量: {len(df_all)}")
    print(f"成功提取数: {extracted_distances.notna().sum()}")
    print(f"面前 (+) 数: {(extracted_distances > 0).sum()}")
    print(f"面后 (-) 数: {(extracted_distances < 0).sum()}")
    print(f"缺失 (NaN) 数: {extracted_distances.isna().sum()}")
    
    # 5. 保存独立的新文件
    output_filename = script_dir / '7303_xyzet_rotated-face_distance.csv'
    df_new.to_csv(output_filename, index=False, encoding='utf-8')
    print(f"\n清洗完成！结果已保存至: {output_filename}")

    # --- 对比展示 ---
    print("\n--- 复杂混乱用例解析演示 ---")
    demo_data = pd.DataFrame({
        '位置(原)': df_all['位置'],
        '巷道(原)': df_all['巷道'],
        '提取结果': df_new['dist_to_face_m']
    })
    # 筛选出一些典型的乱写数据展示一下
    mask = demo_data['位置(原)'].astype(str).str.contains('面前') | \
           demo_data['巷道(原)'].astype(str).str.contains('工作面')
    print(demo_data[mask].head(10).to_string())


if __name__ == "__main__":
    main()