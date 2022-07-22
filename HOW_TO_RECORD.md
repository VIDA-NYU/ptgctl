# How to make a recording with the hololens

## Preparing the hololens

 - turn on the hololens 2 (the one that is closest to the door usually)
 - login using the password `research`
 - Open the application menu and select `HLSStreamerExample`
 - For the alpha version of the app it shows a white grid lattice over objects in the scene and a 
   profiler that reports what it thinks the FPS is

Once you see that screen, the hololens should be uploading.

## Testing (optional)
If you want to test out the video for a certain camera, you can do:
```bash
ptgctl display imshow main  # main camera
ptgctl display imshow gll+grr  # left-most and right-most cameras
```
and this will show the video streams.

## **Now once you're ready to begin recording:**

```bash

# start a recording
ptgctl recordings start  
# will generate an ID using the date+time

# or instead, you give it a memorable name
ptgctl recordings start my-special-recording

# while it's running.... you can run this to check the progress

# get the current recording ID (if there is none it will show nothing)
ptgctl recordings current
ptgctl recordings current --info  # with stats

# 5 mins later....

# ok I'm all done recording,,,, now what?

# stop a recording
ptgctl recordings stop

# get your recording stats
ptgctl recordings get my-special-recording
```

Remember to keep the Hololens charged, so please plug it in when you're done! :)

## Getting exported videos & JSON

```bash
ptgctl recordings static my-special-recording main.mp4  # main camera
ptgctl recordings static my-special-recording gll.mp4  # grey left-rear camera
ptgctl recordings static my-special-recording glf.mp4  # grey left-front camera
ptgctl recordings static my-special-recording grf.mp4  # grey right-front camera
ptgctl recordings static my-special-recording grl.mp4  # grey right-rear camera
ptgctl recordings static my-special-recording depthlt.mp4  # depth camera

ptgctl recordings static my-special-recording mic0.wav  # microphone

ptgctl recordings static my-special-recording eye.json
ptgctl recordings static my-special-recording hand.json

ptgctl recordings static my-special-recording imuaccel.json
ptgctl recordings static my-special-recording imugyro.json
ptgctl recordings static my-special-recording imumag.json
ptgctl recordings static my-special-recording gllCal.json
ptgctl recordings static my-special-recording glfCal.json
ptgctl recordings static my-special-recording grfCal.json
ptgctl recordings static my-special-recording grlCal.json
ptgctl recordings static my-special-recording depthltCal.json
```

