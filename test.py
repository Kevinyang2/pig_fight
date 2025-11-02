import importlib

def check_flash_attention():
    try:
        flash_attn = importlib.import_module("flash_attn")
        print("✅ 已安装 flash-attn")
        print("📦 版本:", flash_attn.__version__)
        return True
    except ImportError:
        print("❌ 未安装 flash-attn，将使用 PyTorch 默认注意力实现")
        return False

if __name__ == "__main__":
    installed = check_flash_attention()

    if installed:
        try:
            # 尝试导入具体函数
            from flash_attn import flash_attn_func
            print("🔍 flash_attn_func 可用，可以进行高效注意力计算")
        except ImportError:
            print("⚠️ flash-attn 已安装，但没有找到 flash_attn_func，可能版本不完整")
