import time
import dfxdemo.dfxdemo
import asyncio

asyncio.run(dfxdemo.dfxdemo.run_measurements("config1.json",2,120, 1))
time.sleep(600)