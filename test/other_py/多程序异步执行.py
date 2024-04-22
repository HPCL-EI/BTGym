import asyncio


async def calculate_square(number):
    print(f"Calculating square of {number}...")
    square = number ** 2
    print(f"Square of {number} is {square}")


async def main():
    # 计算1加到100000
    total = sum(range(1, 100001))


    # 异步执行计算平方的任务
    tasks = [calculate_square(number) for number in range(1, 5)]
    await asyncio.gather(*tasks)
    print(f"Total sum is {total}")

if __name__ == "__main__":
    asyncio.run(main())
