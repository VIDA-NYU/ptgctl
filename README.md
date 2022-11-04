# ptgctl

A Python Library and Command Line tool for interacting with the PTG API and data streams.

### For how to record, see [this](HOW_TO_RECORD.md).

## Install

```bash
# install
pip install 'git+ssh://git@github.com/VIDA-NYU/ptgctl.git[all]'
```

If you want to install and make additions to the package, then you should install like this:
```bash
# install
git clone git+ssh://git@github.com/VIDA-NYU/ptgctl.git
cd ptgctl
pip install -e '.[all]'
```

## CLI Usage


### Recordings

```bash
# list available recordings
ptgctl recordings ls

# list all recordings with their stats (e.g. timestamps, size, etc.)
ptgctl recordings ls --info

# get the stats for recording coffee-test-1
ptgctl recordings get coffee-test-1

# start recording
ptgctl recordings start my-recording-1
# get the current recording
ptgctl recordings current
ptgctl recordings current --info  # get it with stats 
# stop recording
ptgctl recordings stop
```

Access recording exports: e.g.
```bash
ptgctl recordings static my-recording-1 main.mp4  # main camera
ptgctl recordings static my-recording-1 gll.mp4   # grey left-rear camera
ptgctl recordings static my-recording-1 depthlt.mp4  # depth camera
ptgctl recordings static my-recording-1 mic0.wav   # microphone
ptgctl recordings static my-recording-1 eye.json   # eye tracking
ptgctl recordings static my-recording-1 hand.json  # hand tracking
ptgctl recordings static my-recording-1 pointcloud.json  # point cloud 
ptgctl recordings static my-recording-1 depthltCal.json  # depth calibration data
...
```


### Available Streams
get/set details about a stream. This represents the metadata associated with a single data stream (e.g. a camera, accelerometer, or a microphone)

TODO: `ptgctl streams ls` only shows streams if they have uploaded data. This means `ptgctl streams new $stream_id` won't be shown

```bash
stream_id=main   # main camera
stream_id=glf    # grayscale, left-front camera
stream_id=mic0   # mic 0
# ...

# get list of stream names
ptgctl streams ls

# get list of stream names with stream info
ptgctl streams ls --info

# get info about a stream
ptgctl streams get $stream_id

# create a stream
ptgctl streams new $stream_id --desc "some description"

# update a stream
ptgctl streams update $stream_id --desc "some new description" --some-metadata blah --something-else 5

# delete a stream
ptgctl streams delete $stream_id
```

### Available Recipes
get/set details about a recipe.

> TODO: Let's discuss the information we want to store in each step. 

> TODO: Add example with already parsed steps

```bash
# get list of recipe names
ptgctl recipes ls

# get info about a recipe
ptgctl recipes get $recipe_id

# create a recipe TODO: We don't have automatic recipe parsing setup yet.
ptgctl recipes new $recipe_id --title "Mug Cake" --text "Some recipe to parse..."

# get update a recipe 
ptgctl recipes update $recipe_id --title "Mug Cake wooooo"

# get delete a recipe
ptgctl delete-recipe $recipe_id
```


### Sessions
get/set details about a session. This is meant to be a context that the step tracking and application state can exist in.



```bash
# get list of sessions names
ptgctl sessions ls
# get all sessions and their info/metadata
ptgctl sessions ls --info

# get info about a session
ptgctl sessions get $session_id

# create a session
ptgctl sessions new --desc "some description"

# get update a session
ptgctl sessions update $session_id --desc "some new description" --some-metadata blah --something-else 5

# get delete a session
ptgctl sessions delete $session_id
```

### Data Streams

There are data stream methods, however they are not super useful on their own via the CLI. See the [Displaying Data](#displaying-data) section for ways to utilize this from the CLI.

### Creating Data

```bash
# upload a video stream from your webcam
ptgctl mock video $stream_id
# if your webcam isn't default, you can pass a different index
ptgctl mock video $stream_id --src 1
# there's also nothing stopping you from passing a video path
ptgctl mock video $stream_id --src some/video.mp4
# you can also simulate the left/right greyscale cameras
ptgctl mock video $stream_id --pos 1  # glr - left rear
ptgctl mock video $stream_id --pos 2  # glf - left front
ptgctl mock video $stream_id --pos 3  # grf - right front
ptgctl mock video $stream_id --pos 4  # grr - right rear


# watch a video stream from the very beginning
ptgctl display imshow $stream_id --last-entry-id 0

# if the video stream was stored as a hololens frame, you can parse it with this flag
ptgctl display imshow $stream_id --raw-holo
```

You can also record audio and upload it.

```bash
stream_id=mic0

# upload an audio stream
ptgctl mock audio $stream_id
```

### Displaying Data

```bash
# watch a video stream
ptgctl display imshow $stream_id
# watch a video stream from the very beginning
ptgctl display imshow $stream_id --last-entry-id 0

# if the video stream was stored as a hololens frame, you can parse it with this flag
ptgctl display imshow $stream_id --raw-holo
```

```bash
stream_id=mic0

# play an audio stream starting now
ptgctl display audio $stream_id

# play an audio stream starting from the beginning of the stream
ptgctl display audio $stream_id --last-entry-id 0
```


### Choosing the server

Right now, you can select your server like this:

> I have the default set to the local server for now. Use `--nolocal` for the remote server.

```bash
# by default, it will use the live main server
ptgctl streams
# but you can point it to the server on your computer (localhost:7890) like this
ptgctl streams --local
# if the default happens to be the local server, then you can select the remote server like this:
ptgctl streams --nolocal
# or if you need to give it the url to another machine
ptgctl streams --url 192.168.1.17:7890
```


## Python Usage

The usage in Python is fundamentally the same as the CLI, but using Python syntax. 

The CLI uses [fire](https://google.github.io/python-fire/guide/) to generate a command line interface 
from the `API` class, so anything you do with the CLI is fundamentally a python instance method or property of 
the `API()` object.

```python
import ptgctl

api = API()

streams = api.streams.ls()
stream = api.streams.get(stream_id)
api.streams.update(
    stream_id, desc='some new description', 
    some_metadata='blah', something_else=5)

recipes = api.recipes.ls()
recipe = api.recipes.get(recipe_id)
# etc...
```

### Accessing Data

Here's how you can display images using both sync and async code.

```python

import io
import cv2
from PIL import Image

# synchronous, polling

def imshow(api, stream_id, **kw):
      while True:
          for sid, ts, data in api.data(stream_id):
              # load image and display
              im = np.array(Image.open(io.BytesIO(data)))
              im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
              cv2.imshow(sid, im)
          # this is needed for imshow to work
          if cv2.waitKey(1) & 0xFF == ord('q'):
              break

imshow(api, 'main')

# async, streaming

async def imshow_async(api, stream_id, **kw):
    async with api.data_pull_connect(stream_id, **kw) as ws:
        while True:
            for sid, ts, data in await ws.recv_data():
                # load image and display
                im = np.array(Image.open(io.BytesIO(data)))
                im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
                cv2.imshow(sid, im)
            # this is needed for imshow to work
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

import asyncio
asyncio.run(imshow_async(api, 'main'))
```


And the same applies for other data streams:

```python

import io
import cv2
from PIL import Image

# synchronous, polling

def pull(api, stream_id, **kw):
      while True:
          for sid, ts, data in api.data(stream_id):
              print(sid, ts)

pull(api, 'gyro')

# async, streaming

async def pull_async(api, stream_id, **kw):
    async with api.data_pull_connect(stream_id, **kw) as ws:
        while True:
            for sid, ts, data in await ws.recv_data():
                print(sid, ts)

import asyncio
asyncio.run(pull_async(api, 'gyro'))
```


## API Docs

Full API Documentation can be generated using 

```bash
# for development
cd docs/
sphinx-autobuild . _build/html --watch ../ptgctl
```

```bash
# for viewing the compiled 
cd docs/
make html  # optional - to build the latest version.
python -m http.server -d _build/html/
```


## TODO
 - Recipe Upload
   - (api) support uploading a full recipe text and parsing text into steps
     - launch background nlp task on upload
   - Question: What information does the recipe parsing need?

 - Stream Update
   - (api) support both metadata merging and overwriting

 - Data Streams
   - (ctl) support dumping to file
   - (ctl) synchonous websocket generator?

 - CLI Config
   - (ctl) persist url change (switch from remote/local and remember)

 - Tools
   - Display
     - show example with both upload and download at the same time
