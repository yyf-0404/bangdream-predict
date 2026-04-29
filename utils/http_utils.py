# utils/http_utils.py
import time
import requests
import json
import os

def get_json_with_retry(url, max_retries=5):
    """带重试机制的 JSON 获取函数"""
    for i in range(max_retries):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            if i == max_retries - 1:
                print(f"请求失败 ({max_retries}次重试): {e}")
                return None
            print(f"第{i+1}次尝试失败，等待重试...")
            time.sleep(1)
    return None

def save_json(data, filepath, indent=2, ensure_ascii=False):
    """保存 JSON 文件，递归创建目录"""
    try:
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
        print(f"JSON已保存到: {filepath}")
        return True
    except Exception as e:
        print(f"保存JSON失败: {e}")
        return False