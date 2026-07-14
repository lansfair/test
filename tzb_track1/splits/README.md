# 天智杯车辆检测数据划分

该目录保存 `car_det_train.rar` 的固定清理与划分结果。生成参数为：

```text
随机种子: 3407
验证集比例: 0.2
划分方式: SHA-256 去重后的随机多标签分层
平衡目标: 每类出现图像数、实例数、图像尺寸、目标密度
```

数据审计结果：

```text
原始 TIFF/XML 对: 9445
坏图: 3（85.tif、3057.tif、3250.tif）
字节完全相同的重复影像: 133
有效唯一图像: 9309
训练集: 7447
验证集: 1862
```

`train.txt` 和 `val.txt` 是训练实际使用的列表。`drop_invalid.txt` 保存坏图；
`drop_exact_duplicate.txt` 保存被排除的重复影像；`split_manifest.json` 保存
完整类别统计；`sample_validation.csv` 保存逐图校验结果。

生成脚本位于 `lansfair/tzb_track1` 仓库：

```text
tools/dataset_converters/build_tianzhibei_random_split.py
```
