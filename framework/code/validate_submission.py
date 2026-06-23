"""提交文件校验工具

本脚本用于在提交天池前检查 `A1.csv`、`A2.csv` 或 `prediction.zip`
是否满足赛题格式要求。它只做格式和合法性检查，不计算线上分数。
"""
import argparse
import io
import os
import sys
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class CheckResult:
    """单项校验结果"""

    name: str
    ok: bool
    messages: List[str]


def _read_csv_from_zip(zip_path: str, name: str) -> pd.DataFrame:
    """从zip包根目录读取指定CSV"""
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open(name) as f:
            content = f.read()
    return pd.read_csv(io.BytesIO(content))


def _load_inputs(args) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """读取A1/A2提交文件"""
    if args.zip_path:
        with zipfile.ZipFile(args.zip_path, "r") as zf:
            names = zf.namelist()
        expected = ["A1.csv", "A2.csv"]
        if sorted(names) != expected:
            raise ValueError(
                f"zip根目录必须只包含 {expected}，当前包含: {names}"
            )
        return (
            _read_csv_from_zip(args.zip_path, "A1.csv"),
            _read_csv_from_zip(args.zip_path, "A2.csv"),
        )

    if not args.a1_path or not args.a2_path:
        raise ValueError("未提供 --zip_path 时，必须同时提供 --a1_path 和 --a2_path")
    return pd.read_csv(args.a1_path), pd.read_csv(args.a2_path)


def check_a1(a1_df: pd.DataFrame, cls_data_path: str) -> CheckResult:
    """校验A1分类提交文件"""
    messages = []
    ok = True

    data = np.load(cls_data_path)
    expected_test_idx = data["test_idx"]
    labels = data["labels"]
    num_classes = int(labels[labels >= 0].max()) + 1

    if list(a1_df.columns) != ["test_idx", "label"]:
        ok = False
        messages.append(f"表头错误，应为 ['test_idx', 'label']，当前为 {list(a1_df.columns)}")

    if len(a1_df) != len(expected_test_idx):
        ok = False
        messages.append(f"行数错误，应为 {len(expected_test_idx)}，当前为 {len(a1_df)}")

    if "test_idx" in a1_df.columns and len(a1_df) == len(expected_test_idx):
        got = a1_df["test_idx"].to_numpy()
        if not np.array_equal(got, expected_test_idx):
            ok = False
            first_bad = int(np.where(got != expected_test_idx)[0][0])
            messages.append(
                f"test_idx顺序错误，首个不一致位置 {first_bad}: "
                f"提交={got[first_bad]}, 期望={expected_test_idx[first_bad]}"
            )

    if "label" in a1_df.columns:
        labels_series = a1_df["label"]
        if labels_series.isna().any():
            ok = False
            messages.append(f"label存在缺失值: {int(labels_series.isna().sum())} 个")
        numeric = pd.to_numeric(labels_series, errors="coerce")
        if numeric.isna().any():
            ok = False
            messages.append("label存在无法转换为整数的值")
        else:
            int_values = numeric.astype(int)
            if not np.allclose(numeric.to_numpy(), int_values.to_numpy()):
                ok = False
                messages.append("label必须是整数类别编号")
            bad_mask = (int_values < 0) | (int_values >= num_classes)
            if bad_mask.any():
                ok = False
                examples = int_values[bad_mask].head(5).tolist()
                messages.append(
                    f"label越界，合法范围为 [0, {num_classes - 1}]，示例: {examples}"
                )

    if ok:
        label_counts = a1_df["label"].value_counts().sort_index().to_dict()
        messages.append(f"A1通过，行数={len(a1_df)}，类别分布={label_counts}")
    return CheckResult("A1.csv", ok, messages)


def _parse_prediction(value) -> List[str]:
    """解析A2 prediction字段"""
    if pd.isna(value):
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def check_a2(a2_df: pd.DataFrame, rec_data_dir: str, topk: int) -> CheckResult:
    """校验A2推荐提交文件"""
    messages = []
    ok = True

    test_df = pd.read_csv(os.path.join(rec_data_dir, "test.csv"))
    item_df = pd.read_csv(os.path.join(rec_data_dir, "item.csv"))
    expected_uids = test_df["uid"].astype(str).tolist()
    candidate_items = set(item_df["iid"].astype(str).tolist())

    if list(a2_df.columns) != ["uid", "prediction"]:
        ok = False
        messages.append(f"表头错误，应为 ['uid', 'prediction']，当前为 {list(a2_df.columns)}")

    if len(a2_df) != len(expected_uids):
        ok = False
        messages.append(f"行数错误，应为 {len(expected_uids)}，当前为 {len(a2_df)}")

    if "uid" in a2_df.columns and len(a2_df) == len(expected_uids):
        got_uids = a2_df["uid"].astype(str).tolist()
        if got_uids != expected_uids:
            ok = False
            first_bad = next(i for i, (a, b) in enumerate(zip(got_uids, expected_uids)) if a != b)
            messages.append(
                f"uid顺序错误，首个不一致位置 {first_bad}: "
                f"提交={got_uids[first_bad]}, 期望={expected_uids[first_bad]}"
            )

    if "prediction" in a2_df.columns:
        bad_len = []
        bad_dup = []
        bad_item = []
        for row_idx, value in enumerate(a2_df["prediction"]):
            items = _parse_prediction(value)
            if len(items) != topk:
                bad_len.append((row_idx, len(items)))
            if len(set(items)) != len(items):
                bad_dup.append(row_idx)
            illegal = [item for item in items if item not in candidate_items]
            if illegal:
                bad_item.append((row_idx, illegal[:3]))

        if bad_len:
            ok = False
            messages.append(f"prediction长度错误，示例: {bad_len[:5]}，要求每行 {topk} 个")
        if bad_dup:
            ok = False
            messages.append(f"prediction存在重复item，示例行: {bad_dup[:5]}")
        if bad_item:
            ok = False
            messages.append(f"prediction存在非法item，示例: {bad_item[:5]}")

    if ok:
        first_items = a2_df["prediction"].head(3).tolist()
        messages.append(
            f"A2通过，行数={len(a2_df)}，候选item数={len(candidate_items)}，示例={first_items}"
        )
    return CheckResult("A2.csv", ok, messages)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="校验A1/A2提交文件格式")
    parser.add_argument("--zip_path", type=str, default="", help="prediction.zip路径")
    parser.add_argument("--a1_path", type=str, default="", help="A1.csv路径")
    parser.add_argument("--a2_path", type=str, default="", help="A2.csv路径")
    parser.add_argument(
        "--cls_data_path",
        type=str,
        default="data/cls_data/A1.npz",
        help="A1 npz数据路径",
    )
    parser.add_argument(
        "--rec_data_dir",
        type=str,
        default="data/rec_data",
        help="A2推荐数据目录",
    )
    parser.add_argument("--topk", type=int, default=10, help="A2推荐列表长度")
    return parser.parse_args()


def main():
    """主入口"""
    args = parse_args()
    try:
        a1_df, a2_df = _load_inputs(args)
        results = [
            check_a1(a1_df, args.cls_data_path),
            check_a2(a2_df, args.rec_data_dir, args.topk),
        ]
    except Exception as exc:
        print(f"[校验失败] {exc}")
        return 1

    all_ok = True
    for result in results:
        status = "通过" if result.ok else "失败"
        print(f"\n[{result.name}] {status}")
        for message in result.messages:
            print(f"  - {message}")
        all_ok = all_ok and result.ok

    if all_ok:
        print("\n提交文件校验通过。")
        return 0
    print("\n提交文件校验失败，请修正后再打包提交。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
