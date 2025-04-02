import asyncio
import aiohttp
import time
import json
from typing import Dict, List
from datetime import datetime

test_n = 50

async def test_single_request(session: aiohttp.ClientSession, event_content: str) -> Dict:
    """测试单个请求"""
    url = "http://localhost:5678/event/event_analysis"
    start_time = time.time()
    try:
        async with session.post(
            url,
            json={"event_content": event_content, "use_cot_model": False}
        ) as response:
            result = await response.json()
            end_time = time.time()
            return {
                "status": response.status,
                "time": end_time - start_time,
                "result": result
            }
    except Exception as e:
        end_time = time.time()
        return {
            "status": 500,
            "time": end_time - start_time,
            "error": str(e)
        }

async def run_concurrent_tests(n: int = 10, event_content: str = None) -> Dict:
    """运行并发测试
    
    Args:
        n: 并发请求数量
        event_content: 测试的事件内容
        
    Returns:
        测试统计信息
    """
    if event_content is None:
        with open("test_data/test.data", "r", encoding="utf-8") as f:
            event_content = f.read()
            
    async with aiohttp.ClientSession() as session:
        # 创建n个并发任务
        tasks = [test_single_request(session, event_content) for _ in range(n)]
        results = await asyncio.gather(*tasks)
        
        # 统计结果
        success_count = sum(1 for r in results if r["status"] == 200)
        error_count = n - success_count
        times = [r["time"] for r in results]
        
        return {
            "total_requests": n,
            "success_count": success_count,
            "error_count": error_count,
            "success_rate": success_count / n * 100,
            "avg_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times),
            "results": results
        }

def save_test_results(results: Dict, filename: str = None):
    """保存测试结果到文件
    
    Args:
        results: 测试结果字典
        filename: 文件名，如果为None则使用时间戳生成
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_results_{timestamp}.json"
    
    # 创建results目录（如果不存在）
    import os
    os.makedirs("logs", exist_ok=True)
    
    # 保存完整结果
    full_path = os.path.join("logs", filename)
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 保存统计信息到单独的文本文件
    stats_path = os.path.join("logs", f"stats_{timestamp}.txt")
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write("测试统计:\n")
        f.write(f"总请求数: {results['total_requests']}\n")
        f.write(f"成功请求: {results['success_count']}\n")
        f.write(f"失败请求: {results['error_count']}\n")
        f.write(f"成功率: {results['success_rate']:.2f}%\n")
        f.write(f"平均响应时间: {results['avg_time']:.2f}秒\n")
        f.write(f"最短响应时间: {results['min_time']:.2f}秒\n")
        f.write(f"最长响应时间: {results['max_time']:.2f}秒\n")
    
    print(f"\n测试结果已保存到:")
    print(f"完整结果: {full_path}")
    print(f"统计信息: {stats_path}")

async def main():
    # 测试并发性能
    print("开始并发测试...")
    test_results = await run_concurrent_tests(test_n)
    
    # 保存测试结果
    save_test_results(test_results)
    
    # 打印简要统计信息
    print("\n测试统计:")
    print(f"总请求数: {test_results['total_requests']}")
    print(f"成功请求: {test_results['success_count']}")
    print(f"失败请求: {test_results['error_count']}")
    print(f"成功率: {test_results['success_rate']:.2f}%")
    print(f"平均响应时间: {test_results['avg_time']:.2f}秒")
    print(f"最短响应时间: {test_results['min_time']:.2f}秒")
    print(f"最长响应时间: {test_results['max_time']:.2f}秒")

if __name__ == "__main__":
    asyncio.run(main())