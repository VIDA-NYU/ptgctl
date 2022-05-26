'''This is a redis class that can be used for apps that want to connect directly to the redis instance


WIP obviously
'''
import os
import redis.asyncio as aioredis


DEFAULT_URL = os.getenv('REDIS_URL') or 'redis://127.0.0.1:6379'
DEFAULT_BLOCK = 1000

class RedisAPI:
    def __init__(self, url=None):
        self.url = url or DEFAULT_URL

    async def connect(self):
        self.redis = await aioredis.from_url(self.url)
        return self

    async def __aenter__(self):
        return await self.connect()




    async def latest(self, sid, count=1):
        '''Gets the latest'''
        streams = await self.redis.xrevrange(sid, count=count)
        return streams[0][1] if streams else []

    async def next_after(self, sid, last, count=1, block=DEFAULT_BLOCK):
        '''Get the next item after.'''
        streams = await self.redis.xread({sid:last}, count, block=block)
        return streams[0][1] if streams else []

    async def get(self, sid, last='*', count=1, block=DEFAULT_BLOCK):
        '''Get entry.'''
        if last == '*':
            return await self.latest(sid, count)
        return await self.next_after(sid, last, count, block=block)

    async def range(self, sid, start, end, count=1):
        '''Gets the between two times'''
        raise NotImplementedError
        # streams = await self.redis.xrevrange(sid, count=count)
        # return streams[0][1] if streams else []
    
    async def iterrange(self, sid, start, end, count=1, reverse=False):
        '''Gets the between two times'''
        raise NotImplementedError