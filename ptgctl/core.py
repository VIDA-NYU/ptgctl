import os
import sys
import json
import time
import asyncio
import requests
from urllib.parse import urlencode
from . import util
from .util import color

log = util.getLogger(__name__, level='debug')


URL = 'https://eng-nrf233-01.engineering.nyu.edu/ptg/api'
LOCAL_URL = 'http://localhost:7890'

class API:
    '''The PTG API. 
    
    This let's you operate PTG from Python or CLI.
    '''
    _TOKEN_CACHE = '~/.ptg'
    _token = None

    def __init__(self, url: str=None, token=None, username: str=None, password: str=None, local: bool=True):
        url = url or (LOCAL_URL if local else URL)
        # get url and make sure that it has a protocol on it.
        self.url = url = url.rstrip('/')
        secure = True if url.startswith('https://') else False if url.startswith('http://') else None
        uri = url.split('://', 1)[1] if secure is not None else url

        # get websocket url
        self._wsurl = f"ws{'s' if secure else ''}://{uri}"

        self.sess = requests.Session()

        # check token login
        self._TOKEN_CACHE = token_cache = os.path.expanduser(self._TOKEN_CACHE) if self._TOKEN_CACHE else None
        if token:
            self.token = token
        elif username or password:
            self.login(username, password)
        elif token_cache and os.path.isfile(token_cache):
            with open(token_cache, 'r') as f:
                self.token = f.read().strip()
        # elif TOKEN:
        #     self.token = util.Token(TOKEN)
        # elif ask_login:
        #     raise NotImplementedError
        # elif require_login:
        #     raise ValueError("Could not log in.")

    @property
    def token(self) -> util.Token:
        '''The user JWT token used for making authenticated requests.'''
        return self._token

    @token.setter
    def token(self, token):
        self._token = util.Token(token)
    

    def login(self, username: str, password: str):
        '''Login using a username and password to retrieve a token.'''
        assert username and password, "I assume you don't want an empty username and password"
        # get token
        r = self.sess.post(url=f'{self.url}/token',data={'username': username, 'password': password})
        # store token
        self.token = r.json()['access_token']
        if self._TOKEN_CACHE:
            with open(self._TOKEN_CACHE, 'w') as f:
                f.write(str(self.token))

    def logout(self):
        '''Logout and discard the token.'''
        self.token = None
        if os.path.isfile(self._TOKEN_CACHE):
            os.remove(self._TOKEN_CACHE)

    # make a request

    def _headers(self, headers: dict=None, **kw) -> dict:
        return {'Authorization': f'Bearer {self.token}', **(headers or {}), **kw}

    def _do(self, method: str, *url_parts, headers: dict=None, params: dict=None, raises: bool=True, **kw) -> requests.Response:
        '''Generic request wrapper.'''
        url = os.path.join(self.url, *map(str, url_parts))
        headers = self._headers(headers)
        if params:
            kw['params'] = {k: v for k, v in params.items() if k and v is not None}

        # make the request and time it
        log.info('request: %s %s', method, url)
        t0 = time.time()
        r = self.sess.request(method, url, headers=headers, **kw)
        log.info('took %.3g secs', time.time() - t0)
        if raises:
            r.raise_for_status()
        return r

    def _get(self, *a, **kw) -> requests.Response: return self._do('GET', *a, **kw)
    def _put(self, *a, **kw) -> requests.Response: return self._do('PUT', *a, **kw)
    def _post(self, *a, **kw) -> requests.Response: return self._do('POST', *a, **kw)
    def _delete(self, *a, **kw) -> requests.Response: return self._do('DELETE', *a, **kw)

    def _ws(self, *url_parts, headers: dict=None, connect_kwargs: dict=None, **params):
        # import websockets
        params = urlencode(params) if params else None
        url = os.path.join(self._wsurl, *map(str, url_parts)) + (f'?{params}' if params else '')
        log.info('websocket connect: %s', url)
        # return websockets.connect(url, extra_headers=self._headers(headers), **(connect_kwargs or {}))
        return WebsocketStream(url, extra_headers=self._headers(headers), **(connect_kwargs or {}))

    def ping(self, error=False):
        if error:
            return self._get('ping/error').json()
        return self._get('ping').json()


    # manage streams

    class streams(util.Nest):
        '''Data Stream metadata.'''
        def ls(self, info: bool=None) -> list:
            '''Get all streams'''
            return self._get('streams', params={'info': info}).json()

        def get(self, id: str, report_error: bool=None) -> dict:
            '''Get a stream.
            
            Arguments:
                id (str): The stream ID.
            '''
            return self._get('streams', id, params={'report_error': report_error}).json()

        def new(self, id: str, desc: str=None, *, override: bool=None, max_len: int=None, **meta):
            '''Create a stream.
            
            Arguments:
                id (str): The stream ID.
                desc (str): The stream description.
                override (bool): Whether to overwrite an existing stream. Otherwise it will throw an error.
                **meta: Any arbitrary metadata to attach to the stream.
            '''
            return self._put('streams', id, params={
                'desc': desc,
                'override': override,
                'meta': json.dumps(meta),
                'max_len': max_len,
            }).json()

        def delete(self, id: str) -> bool:
            '''Delete a stream.
            
            Arguments:
                id (str): The stream ID.
            '''
            return self._delete('streams', id).json()


    # recipes

    class recipes(util.Nest):
        '''Manage recipes.'''
        def ls(self) -> list:
            '''Get all recipes.'''
            return self._get('recipes').json()

        def get(self, id: str) -> dict:
            '''Get a recipe by ID.
            
            Arguments:
                id (str): The recipe ID.
            '''
            return self._get('recipes', id).json()

        def new(self, title: str=None, text: str=None, steps: list=None):
            '''Create a recipe.
            
            Arguments:
                id (str): The recipe ID.
                title (str): The recipe title.
                text (str): The full to-be-parsed text of the recipe. Use ``text=True`` (``--text``) to read the text from stdin.
                steps (list[dict]): The already-parsed recipe steps. An alternative to ``text``.
            '''
            if text is True:
                text = sys.stdin.read()
            return self._post('recipes', json=util.filternone({
                'title': title,
                'text': text,
                'steps': steps,
            })).json()

        def update(self, id: str, title: str=None, text: str=None, steps: list=None) -> bool:
            '''Update a recipe.
            
            Arguments:
                id (str): The recipe ID.
                title (str): The recipe title.
                text (str): The full to-be-parsed text of the recipe. Use ``text=True`` (``--text``) to read the text from stdin.
                steps (list[dict]): The already-parsed recipe steps. An alternative to ``text``.
            '''
            if text is True:
                text = sys.stdin.read()
            return self._put('recipes', id, json=util.filternone({
                'title': title,
                'text': text,
                'steps': steps,
            })).json()

        def delete(self, id: str) -> bool:
            '''Delete a recipe.
            
            Arguments:
                id (str): The recipe ID.
            '''
            return self._delete('recipes', id).json()


    # session

    class sessions(util.Nest):
        '''User session management.'''
        def ls(self) -> list:
            '''Get all sessions.'''
            return self._get('sessions').json()

        def get(self, id: str) -> dict:
            '''Get a session by ID.
            
            Arguments:
                id (str): The session ID.
            '''
            return self._get('sessions', id).json()

        def new(self, recipe_id: str, step_index: int=None):
            '''Create a session.
            
            Arguments:
                id (str): The session ID.
            '''
            return self._post('sessions', json=util.filternone({
                'recipe_id': recipe_id,
                'step_index': step_index,
            })).json()

        def update(self, id, recipe_id=None, step_index=None) -> bool:
            '''Update a session.
            
            Arguments:
                id (str): The session ID.
            '''
            return self._put('sessions', id, json=util.filternone({
                'recipe_id': recipe_id,
                'step_index': step_index,
            })).json()

        def delete(self, id: str) -> bool:
            '''Delete a session.
            
            Arguments:
                id (str): The session ID.
            '''
            return self._delete('sessions', id).json()

        def step(self, id: str) -> int:
            '''Get the current step.'''
            return self._get('sessions', id, 'step').json()

        def update_step(self, id: str, step: int=None) -> bool:
            '''Set the current step.'''
            return self._put('sessions', id, 'step', step).json()


    # data

    def data(self, stream_id: str, **kw) -> dict:
        '''Get the latest data
        
        Arguments:
            id (str): The stream ID.
            **params: url query params for request.
        '''
        r = self._get('data', stream_id, params=kw)
        return util.unpack_entries(json.loads(r.headers['entry-offset']), r.content)

    def upload_data(self, stream_id: str, data: dict, **kw) -> dict:
        '''Upload a data packet. Not Implemented.
        
        Arguments:
            id (str): The stream ID.
            **params: url query params for request.
        '''
        if not isinstance(data, (list, tuple)):
            data = [data]
        
        if isinstance(stream_id, (list, tuple)):
            stream_ids = stream_id
            stream_id = '*'
        else:
            stream_ids = [stream_id] * len(data)
        return self._post('data', stream_id, params=kw, files=dict(zip(stream_ids, data))).json()

    # data over async websockets

    def data_pull_connect(self, stream_id: str, **kw) -> 'WebsocketStream':
        '''Connect to the server and send data over an asynchronous websocket.
        
        .. code-block:: python

            async with api.data_pull_connect(stream_id, **kw) as ws:
                while True:
                    for sid, ts, data in await ws.aread():
                        img = np.array(Image.open(io.BytesIO(data)))  # rgb array
        '''
        return self._ws('data', stream_id, 'pull', **kw)

    def data_push_connect(self, stream_id: str, **kw) -> 'WebsocketStream':
        '''Connect to the server and send data over an asynchronous websocket.
        
        .. code-block:: python

            async with api.data_push_connect(stream_id, **kw) as ws:
                while True:
                    img = get_some_image()

                    # write image to jpg
                    output = io.BytesIO()
                    pil.fromarray(img).save(output, format='jpeg')
                    data = output.getvalue()

                    # TODO: support timestamps + multiple sid
                    await ws.awrite(data)
        '''
        return self._ws('data', stream_id, 'push', **kw)

    # recording TODO

    def start_recording(self, stream_id: str):
        '''Start a recording. Not Implemented'''
        raise NotImplementedError

    def stop_recording(self, stream_id: str):
        '''Stop a recording. Not Implemented'''
        raise NotImplementedError

    # tools

    @util.bound_module
    def display(self) -> util.BoundModule:  # lazy import and bind
        from .tools import display
        return display

    @util.bound_module
    def mock(self) -> util.BoundModule:  # lazy import and bind
        from .tools import mock
        return mock




class WebsocketStream:
    '''Encapsulates a websocket stream to read/write in the format that the server sends it (json offsets, bytes, json, bytes, etc.)'''
    connect = ws = None
    def __init__(self, *a, **kw):
        self.a, self.kw = a, kw

    async def __await__(self):
        await asyncio.sleep(1e-6)
        return await self.connect

    async def __aenter__(self):
        import websockets
        self.connect = websockets.connect(*self.a, **self.kw)
        self.ws = await self.connect.__aenter__()
        await asyncio.sleep(1e-6)
        return self
    
    async def __aexit__(self, *a):
        await self.connect.__aexit__(*a)
        self.connect = self.ws = None
        await asyncio.sleep(1e-6)

    async def recv_data(self):
        await asyncio.sleep(1e-6)
        offsets = json.loads(await self.ws.recv())
        content = await self.ws.recv()
        return util.unpack_entries(offsets, content)

    async def send_data(self, data, batch=False, ack=False):
        offsets, entries = util.pack_entries([data])
        if batch:
            await self.ws.send(','.join(map(str, offsets)))
        await self.ws.send(bytes(entries))
        if ack:
            await self.ws.recv()  # ack
        await asyncio.sleep(1e-6)




# This just implements a few command line features that we may not necessarily want from the python side
class CLI(API):
#     '''

#       ___           ___           ___           ___           ___           ___   
#      /\  \         /\  \         /\  \         /\  \         /\  \         /\__\  
#     /::\  \        \:\  \       /::\  \       /::\  \        \:\  \       /:/  /  
#    /:/\:\  \        \:\  \     /:/\:\  \     /:/\:\  \        \:\  \     /:/  /   
#   /::\~\:\  \       /::\  \   /:/  \:\  \   /:/  \:\  \       /::\  \   /:/  /    
#  /:/\:\ \:\__\     /:/\:\__\ /:/__/_\:\__\ /:/__/ \:\__\     /:/\:\__\ /:/__/     
#  \/__\:\/:/  /    /:/  \/__/ \:\  /\ \/__/ \:\  \  \/__/    /:/  \/__/ \:\  \     
#       \::/  /    /:/  /       \:\ \:\__\    \:\  \         /:/  /       \:\  \    
#        \/__/     \/__/         \:\/:/  /     \:\  \        \/__/         \:\  \   
#                                 \::/  /       \:\__\                      \:\__\  
#                                  \/__/         \/__/                       \/__/  

#     '''
    def _do(self, *a, **kw):
        # make sure we're logged in. If not, prompt
        if not self.token:
            self.login()
        return super()._do(*a, **kw)

    def login(self, username=None, password=None):
        # prompt for missing username/password
        if not username or not password:
            log.warning("You're not logged in:")
            if not username:
                username = input(color("What is your username? ", 'purple', 1))
            if not password:
                import getpass
                password = getpass.getpass(color("What is your password? ", 'purple', 1))
        return super().login(username, password)




def main():
    '''Create an auto-magical CLI !! really that's it.'''
    from .cli_format import yamltable, indent  # installs a formatter for fire
    import fire
    try:
        fire.Fire(CLI)
    except requests.HTTPError as e:
        r = e.response
        if 'application/json' in r.headers.get('Content-Type'):
            content = r.json()
            content = yamltable(content)
        else:
            content = r.content
        # content = r.json() if 'application/json' in r.headers.get('Content-Type') else r.content
        log.error('Error: %d\nContent: \n%s', r.status_code, color(indent(f'{content}', 2), 'red', 1))
        sys.exit(1)



if __name__ == '__main__':
    main()
