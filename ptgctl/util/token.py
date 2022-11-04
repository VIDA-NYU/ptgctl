import json
import base64
import datetime


class Token(dict):
    '''Wraps a JWT token and handles parsing the token information and checking its expiration.
    
    .. code-block:: python

        token = Token(token_str)

        if token:
            print(f'Your token is valid for {token.time_left.total_seconds():.0f} secs.')
            print(token)
            requests.get('/something', headers={'Authorization': f'Bearer {token}'})
        else:
            print('token is empty or has expired.')
    '''
    min_time_left = 10  # seconds
    def __init__(self, token=None):
        self.token = token = str(token or '')
        self.header, self.data, self.signature = jwt_decode(token) if token else ({}, {}, '')
        super().__init__(self.data)

        expires = self.get('exp')
        self.expires = datetime.datetime.fromtimestamp(expires) if expires else None
        self._min_time_left = datetime.timedelta(seconds=self.min_time_left)

    def __repr__(self):
        '''Show the token information, including expiration and data payload.'''
        if self.token is None:
            return 'Token(None)'
        return 'Token(time_left={}, {})'.format(self.time_left, super().__repr__())

    def __str__(self):
        '''Get the token as a string (can be passed in an Authorization header)'''
        return str(self.token or '')

    def __bool__(self):
        '''Check if the token exists and is not expired.'''
        return bool(self.token) and self.time_left > self._min_time_left

    @property
    def time_left(self) -> datetime.timedelta:
        '''Gets the amount of time left, as a datetime.timedelta object.'''
        return self.expires - datetime.datetime.now() if self.expires else datetime.timedelta(seconds=0)

    @classmethod
    def from_cookiejar(cls, jar, domain, name):
        domain = domain.split('://')[-1].split('/')[0].split(':', 1)[0]
        cs = jar._cookies
        if domain not in cs and f'{domain}.local' in cs:
            domain = f'{domain}.local'
        paths = cs.get(domain) or {}    
        for path, cookies in paths.items():
            if name in cookies:
                token = cookies[name].value.strip('"')
                if token and token.startswith('Bearer '):
                    token = token.split(' ', 1)[-1]
                return cls(token)



def partdecode(x):
    '''Decode the token base64 string part as a dictionary. Split the token using ``'.'`` first.'''
    return json.loads(base64.b64decode(x + '===').decode('utf-8'))

# def partencode(x):
#     '''Encode the token dictionary payload as a base64 string.'''
#     return base64.b64encode(json.dumps(x).encode('utf-8')).decode('utf-8')

def jwt_decode(token):
    '''Decode a token into it's parts and parse the header, payload, and signature.'''
    header, data, signature = token.split('.')
    return partdecode(header), partdecode(data), signature

# def jwt_encode(header, data, signature=None):
#     '''Take a token's parts and convert it to a token. This is only really used for mocking a token for testing purposes.'''
#     header, data = partencode(header), partencode(data)
#     signature = signature or hash(str(header+data))  # default to dummy signature
#     return '.'.join((header, data, signature))
