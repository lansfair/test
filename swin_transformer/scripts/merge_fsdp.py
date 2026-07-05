import torch
import torch.distributed.checkpoint as dcp
from pathlib import Path

def merge_fsdp_checkpoint(checkpoint_dir: str, output_path: str = "merged_weights.pt"):
    """
    合并 FSDP 分布式检查点到单个 state_dict
    """
    ckpt_dir = Path(checkpoint_dir)
    
    # 1. 读取元数据
    reader = dcp.FileSystemReader(ckpt_dir)
    metadata = reader.read_metadata()
    
    # 2. 获取 state_dict 元数据
    # 从您打印的内容看，是 state_dict_metadata
    state_dict_meta = metadata.state_dict_metadata
    
    # 3. 构建占位 state_dict
    state_dict = {}
    for key, storage_meta in state_dict_meta.items():
        # 跳过非张量数据（如 iteration）
        if not hasattr(storage_meta, "size"):
            print(f"跳过非张量: {key} (类型: {type(storage_meta).__name__})")
            continue
        
        # 获取 size 和 dtype
        size = storage_meta.size
        dtype = storage_meta.properties.dtype
        
        # 在 CPU 上创建零张量
        state_dict[key] = torch.zeros(size, dtype=dtype, device="cpu")
    
    print(f"共构建 {len(state_dict)} 个张量占位")
    
    # 4. 加载分片数据（填充实际权重）
    dcp.load(state_dict=state_dict, checkpoint_id=ckpt_dir)
    
    # 5. 保存为单个文件
    torch.save(state_dict, ckpt_dir / output_path)
    print(f"✅ 合并成功，已保存至 {output_path}")
    print(f"   State dict 包含 {len(state_dict)} 个 key")
    return state_dict

if __name__ == "__main__":
    # CKPT_DIR = "/mnt/qh2-nas3/00-model/00-wrs/zhejiang_earth_results/swin-distill/30999"
    CKPT_DIR = "/mnt/si000523ygkv/00-model/dinov3-distill-outputs/swin_base_vitl16_ssl_feature_distill_GE+IN22k+ZJSlice1024_16nodes_nowarmup_lowlr/ckpt/30999"
    merge_fsdp_checkpoint(CKPT_DIR, "merged_weights.pt")