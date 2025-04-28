import requests
import json

def get_ask(url, payload):
    response = requests.post(url, json=payload, stream=True)
    if response.status_code == 200:
        last_answer = ""  # 存储上一次的完整回答
        for line in response.iter_lines(decode_unicode=True):
            if line.startswith("data: "):
                data = json.loads(line[6:])
                if "answer" in data and data["answer"] == "<end>":
                    return data
                elif "answer" in data:
                    # 只打印新增的内容
                    new_content = data['answer'][len(last_answer):]
                    if new_content:  # 只有当有新内容时才打印
                        print(new_content, end="", flush=True)
                    last_answer = data['answer']
    else:
        print(f"请求失败，状态码: {response.status_code}")
        print(f"错误信息: {response.text}")

if __name__ == "__main__":
    url = "http://127.0.0.1:5678/course/create_course"
    course = json.dumps("")
    history = []
    while True:
        user_input = input("\n请输入你的需求：")
        payload = {
            "course": course,
            "user_input": user_input,
            "history": history,
            "use_cot_model": False
        }
        try:
            response = get_ask(url, payload)
            if response is None:
                continue_input = input("请求失败，是否继续？（y/n）")
                if continue_input.lower() != "y":
                    break
                continue

            # 新增：如果后端返回了错误，直接打印并跳过
            if isinstance(response, dict) and "error" in response:
                print("后端错误：", response["error"])
                print("Traceback：", response.get("traceback", "无"))
                continue_input = input("请求失败，是否继续？（y/n）")
                if continue_input.lower() != "y":
                    break
                continue

            print(response)  # 打印后端返回内容，便于调试
            is_completed = response["is_completed"]
            if is_completed:
                print("\n课程创建成功")
                break
            course = response["course"]
            course_dict = json.loads(course)
            history = response["history"]
            print("\n当前课程配置：")
            print(json.dumps(course_dict, indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"处理响应时发生错误: {str(e)}")
            continue_input = input("是否继续？（y/n）")
            if continue_input.lower() != "y":
                break