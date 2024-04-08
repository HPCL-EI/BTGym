from openai import OpenAI




class LLMGPT4():
    def __init__(self):
        self.client = OpenAI(
            base_url="https://kapkey.chatgptapi.org.cn/v1",            api_key="sk-YX6Kn2rtcXHmz4en9f6c2931Bf194eD892A09e70E7493fA9"
        )

    def request(self,question):
        completion = self.client.chat.completions.create(
          model="gpt-4-turbo-preview",
          messages=[
            {"role": "system", "content": ""},#You are a helpful assistant.
            {"role": "user", "content": question}
          ]
        )

        return completion.choices[0].message.content


if __name__ == '__main__':
    llm = LLMGPT4()
    answer = llm.request(question="who are you ?")
    print(answer)