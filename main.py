from llm.openrouter import chat

# llm 폴더의 openrouter.py에서 chat 함수를 가져옴
def main():
    system_prompt = "You are a helpful assistant."
    user_prompt = "What is the capital of France?"
    response = chat(system_prompt, user_prompt)
    print(response)


if __name__ == "__main__":
    main()