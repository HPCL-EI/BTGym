from openai import OpenAI




class LLMGPT3():
    def __init__(self):
        self.client = OpenAI(
            base_url="https://api.chatgptid.net/v1",            api_key="sk-l4DF0VWpNxibzbFBAaB94592518343988799DeEbC65cF6Ff"
        )

    def request(self,question):
        completion = self.client.chat.completions.create(
          model="gpt-3.5-turbo",
          messages=[
            {"role": "system", "content": ""},#You are a helpful assistant.
            {"role": "user", "content": question}
          ]
        )

        return completion.choices[0].message.content


if __name__ == '__main__':
    llm = LLMGPT3()
    answer = llm.request(question="who are you ?")
    print(answer)