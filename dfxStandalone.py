import time
import dfxdemo.dfxdemo
import asyncio
import platform


if __name__ == "__main__":
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(dfxdemo.dfxdemo.run_measurements("config1.json",2,120, 1))