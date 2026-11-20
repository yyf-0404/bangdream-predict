# utils/http_utils.py
import time
import requests
import json
import os
import stat
from tempfile import TemporaryDirectory

def get_json_with_retry(url, max_retries=5):
    """带重试机制的 JSON 获取函数"""
    for i in range(max_retries):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            status_code = getattr(getattr(e, "response", None), "status_code", None)
            if status_code is not None and 400 <= status_code < 500 and status_code not in (408, 429):
                print(f"请求失败（不可重试的 HTTP {status_code}）: {e}")
                return None
            if i == max_retries - 1:
                print(f"请求失败 ({max_retries}次重试): {e}")
                return None
            print(f"第{i+1}次尝试失败，等待重试...")
            time.sleep(1)
    return None

def save_json(data, filepath, indent=2, ensure_ascii=False):
    """完整写入后再替换文件，读取方始终能打开一份完整的快照。"""
    try:
        directory = os.path.dirname(filepath) or '.'
        os.makedirs(directory, exist_ok=True)
        with TemporaryDirectory(dir=directory) as pending:
            temporary = os.path.join(pending, 'snapshot.json')
            with open(temporary, 'x', encoding='utf-8') as stream:
                json.dump(data, stream, indent=indent, ensure_ascii=ensure_ascii, allow_nan=False)
            try:
                mode = os.stat(filepath).st_mode
            except FileNotFoundError:
                pass
            else:
                os.chmod(temporary, stat.S_IMODE(mode))
            os.replace(temporary, filepath)
        print(f"JSON已保存到: {filepath}")
        return True
    except Exception as e:
        print(f"保存JSON失败: {e}")
        return False
