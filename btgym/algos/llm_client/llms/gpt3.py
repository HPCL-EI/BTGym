from openai import OpenAI




class LLMGPT3():
    def __init__(self):
        self.client = OpenAI(
            base_url="https://api.chatgptid.net/v1",            api_key="sk-WghHNZ1cSm0nZ3ME79A3B93272F748898409C65aE911Ce19"
        )

    def request(self,message): # question
        completion = self.client.chat.completions.create(
          model="gpt-3.5-turbo",
          # messages=[
          #   {"role": "system", "content": ""},#You are a helpful assistant.
          #   {"role": "user", "content": question}
          # ]
            messages=message
        )

        return completion.choices[0].message.content

    def embedding(self,question):
        embeddings = self.client.embeddings.create(
          model="text-embedding-3-small",
          # model="text-embedding-ada-002",
          input=question
        )

        return embeddings


if __name__ == '__main__':
    llm = LLMGPT3()
    # answer = llm.request(question="who are you,gpt?")
    answer = llm.embedding(question="who are you,gpt?")
    print(answer)

    # llm = LLMGPT3()
    # messages = [{"role": "system", "content": "你现在是很有用的助手！"}]
    # while True:
    #     prompt = input("请输入你的问题:")
    #     messages.append({"role": "user", "content": prompt})
    #     res_msg = llm.request(messages)
    #     messages.append({"role": "assistant", "content": res_msg})
    #     print(res_msg)